from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import matplotlib.colors as mcolors
from datetime import datetime, timezone, timedelta
import uuid
import time
import glob
import csv
from cassandra.cluster import Cluster
from cassandra.cqlengine import connection
from cassandra.cqlengine.models import Model
from cassandra.cqlengine import columns
from cassandra.cqlengine.management import sync_table, create_keyspace_simple

os.environ['CQLENG_ALLOW_SCHEMA_MANAGEMENT'] = '1'

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Configure BraTS2020 data path
BRATS_DATA_PATH = '/app/data/MICCAI_BraTS2020_TrainingData'  # Container path
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Configure Cassandra
CASSANDRA_HOST = ['cassandra']  # Use service name from docker-compose
CASSANDRA_KEYSPACE = 'brain_tumor_app'

# Connect to Cassandra cluster with retry mechanism
max_retries = 5
retry_delay = 5  # seconds
connected = False

for i in range(max_retries):
    try:
        cluster = Cluster(CASSANDRA_HOST)
        session = cluster.connect()
        # Create keyspace if it doesn't exist
        session.execute("""
            CREATE KEYSPACE IF NOT EXISTS %s 
            WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}
            """ % CASSANDRA_KEYSPACE)
        
        connection.setup(
            hosts=CASSANDRA_HOST,
            default_keyspace=CASSANDRA_KEYSPACE,
            protocol_version=4,
            retry_connect=True)
        
        print("Cassandra connection successful after attempt", i+1)
        connected = True
        break
    except Exception as e:
        print(f"Cassandra connection error attempt {i+1}: {e}")
        time.sleep(retry_delay)

if not connected:
    print("Could not connect to Cassandra after multiple attempts. The application will continue but may not function properly.")

# Define storage directories
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
RESULTS_FOLDER = os.path.join(STATIC_FOLDER, 'results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'nii', 'nii.gz'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'img'), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'css'), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'js'), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'lib'), exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Define input image size
IMG_SIZE = 128

# Cassandra Models with MapReduce approach
class Patient(Model):
    __keyspace__ = CASSANDRA_KEYSPACE
    brats20id = columns.Text(primary_key=True)  # Key for MapReduce
    age = columns.Float()
    survival_days = columns.Text()
    extent_of_resection = columns.Text()
    grade = columns.Text()  # HGG or LGG
    # These fields will be used in Reduce operations

class PatientMapping(Model):
    __keyspace__ = CASSANDRA_KEYSPACE
    brats20id = columns.Text(primary_key=True)  # Key for MapReduce
    brats17id = columns.Text()
    brats18id = columns.Text()
    brats19id = columns.Text()
    tcga_id = columns.Text()
    grade = columns.Text()
    # These fields will be used in Map operations to link different IDs

class Prediction(Model):
    __keyspace__ = CASSANDRA_KEYSPACE
    id = columns.UUID(primary_key=True, default=uuid.uuid4)
    brats_id = columns.Text(index=True)  # Key for MapReduce operations
    filename = columns.Text(required=True)
    slice_index = columns.Integer(required=True)
    result_path = columns.Text(required=True)
    created_at = columns.DateTime(default=lambda: datetime.now())
    notes = columns.Text()  # Additional notes about the prediction

# Create tables in Cassandra
sync_table(Patient)
sync_table(PatientMapping)
sync_table(Prediction)

# Dice Coefficient for the model (MapReduce-like aggregation of results)
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4  # Number of segmentation classes (background + tumor parts)
    total_loss = 0
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:, :, :, i])
        y_pred_f = K.flatten(y_pred[:, :, :, i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        total_loss = total_loss + loss if i > 0 else loss
    return total_loss / class_num

# Load the pre-trained model
try:
    model = tf.keras.models.load_model(
        "model/3D_MRI_Brain_tumor_segmentation.h5",
        custom_objects={'dice_coef': dice_coef},
        compile=False
    )
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

# Check valid file format
def allowed_file(filename):
    return '.' in filename and (
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'] or
        filename.rsplit('.', 2)[1] + '.' + filename.rsplit('.', 2)[2] in app.config['ALLOWED_EXTENSIONS']
    )

# MAP: Preprocess MRI image from .nii file
def preprocess_image(file_path, slice_index=60):
    # Map operation: Transform raw data into processable format
    img = nib.load(file_path).get_fdata()
    img = img[:, :, slice_index]  # Get slice
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to (128, 128)
    
    # Avoid division by zero during normalization
    img_max = np.max(img_resized)
    img_normalized = img_resized / img_max if img_max > 0 else img_resized
    
    return img_normalized

# Create custom colormap for tumor segmentation
def create_tumor_segmentation_cmap():
    # Colors for each class:
    # 0: Background (black)
    # 1: NCR/NET (red)
    # 2: ED (yellow)
    # 3: ET (blue)
    colors = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1)]
    return mcolors.ListedColormap(colors)

# REDUCE: Perform tumor prediction on MRI image (combines multiple inputs into single result)
def perform_prediction(flair_path, t1ce_path, slice_index, brats_id=None, notes=None):
    # MAP: Preprocess FLAIR and T1CE images
    X_flair = preprocess_image(flair_path, slice_index)
    X_t1ce = preprocess_image(t1ce_path, slice_index)
    
    # Combine into 2-channel tensor
    X = np.stack([X_flair, X_t1ce], axis=-1)  # (128, 128, 2)
    
    # Format input correctly `(1, 128, 128, 2)`
    X_input = np.expand_dims(X, axis=0)
    
    # Predict - Parallel processing happens inside the model
    pred = model.predict(X_input)
    
    # REDUCE: Get class with highest value
    prediction = np.argmax(pred[0], axis=-1)

    # Create custom colormap
    tumor_cmap = create_tumor_segmentation_cmap()

    # Display results with custom colormap
    plt.figure(figsize=(14, 7))
    
    # Original FLAIR MRI image
    plt.subplot(1, 2, 1)
    plt.imshow(X_flair, cmap='gray')
    plt.title(f"Slice {slice_index} - MRI Image (FLAIR)", fontsize=14)
    plt.axis("off")
    
    # Segmentation image
    plt.subplot(1, 2, 2)
    im = plt.imshow(prediction, cmap=tumor_cmap, vmin=0, vmax=3)
    plt.title(f"Segmentation at slice {slice_index}", fontsize=14)
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['Background', 'NCR/NET', 'ED', 'ET'])
    plt.axis("off")

    # Create unique result filename based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    result_filename = f"result_{timestamp}.png"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    
    # Save result with high quality
    plt.tight_layout()
    plt.savefig(result_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # REDUCE: Store prediction information in database
    filename = os.path.basename(flair_path)
    pred_record = Prediction.create(
        brats_id=brats_id if brats_id else "unknown",
        filename=filename,
        slice_index=slice_index,
        result_path=result_filename,  # Store filename rather than full path
        notes=notes
    )
    
    return result_filename, pred_record.id  # Return result filename and record id

# MAP-REDUCE: Scan BraTS directory and return list of patients
def scan_brats_directory():
    patients = []
    try:
        print(f"BRATS_DATA_PATH: {BRATS_DATA_PATH}")
        print(f"Path exists: {os.path.exists(BRATS_DATA_PATH)}")
        
        # Check for directories matching pattern
        pattern = os.path.join(BRATS_DATA_PATH, "BraTS20*")
        print(f"Search pattern: {pattern}")
        patient_folders = glob.glob(pattern)
        print(f"Number of directories found: {len(patient_folders)}")
        
        # MAP: Process each patient folder
        for folder in patient_folders:
            patient_id = os.path.basename(folder)
            
            # Find .nii or .nii.gz files in directory
            flair_files = glob.glob(os.path.join(folder, "*flair.nii*"))
            t1ce_files = glob.glob(os.path.join(folder, "*t1ce.nii*"))
            
            # REDUCE: Collect only valid patients with required files
            if flair_files and t1ce_files:
                patients.append({
                    'id': patient_id,
                    'flair_path': flair_files[0],
                    't1ce_path': t1ce_files[0]
                })
                print(f"Added patient {patient_id}")
        
        print(f"Total patients found: {len(patients)}")
        return patients
    except Exception as e:
        print(f"Error scanning BRATS directory: {e}")
        import traceback
        traceback.print_exc()
        return []
# Hàm tải dữ liệu từ CSV vào Cassandra
def load_csv_data():
    # Tải dữ liệu survival_info.csv
    survival_path = os.path.join(CSV_PATH, 'survival_info.csv')
    if os.path.exists(survival_path):
        with open(survival_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    Patient.create(
                        brats20id=row['Brats20ID'],
                        age=float(row['Age']) if row['Age'] else None,
                        survival_days=row['Survival_days'],
                        extent_of_resection=row['Extent_of_Resection']
                    )
                except Exception as e:
                    print(f"Lỗi khi tải dữ liệu bệnh nhân {row['Brats20ID']}: {e}")
    
    # Tải dữ liệu name_mapping.csv
    mapping_path = os.path.join(CSV_PATH, 'name_mapping.csv')
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Nếu có BraTS_2020_subject_ID
                    if row['BraTS_2020_subject_ID']:
                        # Kiểm tra xem bệnh nhân đã tồn tại chưa
                        existing_patient = Patient.objects.filter(brats20id=row['BraTS_2020_subject_ID']).first()
                        
                        if existing_patient:
                            # Cập nhật grade nếu bệnh nhân đã tồn tại
                            existing_patient.grade = row['Grade']
                            existing_patient.save()
                        else:
                            # Tạo bệnh nhân mới nếu chưa tồn tại
                            Patient.create(
                                brats20id=row['BraTS_2020_subject_ID'],
                                age=None,  # Không có thông tin tuổi
                                survival_days=None,  # Không có thông tin ngày sống
                                extent_of_resection=None,  # Không có thông tin phẫu thuật
                                grade=row['Grade']  # Chỉ có thông tin grade
                            )
                        
                        # Tạo bản ghi mapping
                        PatientMapping.create(
                            brats20id=row['BraTS_2020_subject_ID'],
                            brats17id=row['BraTS_2017_subject_ID'] if row['BraTS_2017_subject_ID'] else None,
                            brats18id=row['BraTS_2018_subject_ID'] if row['BraTS_2018_subject_ID'] else None,
                            brats19id=row['BraTS_2019_subject_ID'] if row['BraTS_2019_subject_ID'] else None,
                            tcga_id=row['TCGA_TCIA_subject_ID'] if row['TCGA_TCIA_subject_ID'] else None,
                            grade=row['Grade'] if row['Grade'] else None
                        )
                except Exception as e:
                    print(f"Lỗi khi tải dữ liệu mapping {row.get('BraTS_2020_subject_ID', 'unknown')}: {e}")
# --------- ROUTES FOR RETAINED FUNCTIONALITY ---------

# Home page
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/news')
def news():
    return render_template('news.html')
# Thêm route này vào file Python của bạn
@app.route('/load-csv', methods=['GET'])
def load_csv():
    try:
        load_csv_data()
        flash("Dữ liệu CSV đã được tải lên Cassandra thành công!", "success")
    except Exception as e:
        flash(f"Lỗi khi tải dữ liệu CSV: {str(e)}", "danger")
    
    return redirect(url_for('index'))
@app.route('/patients')
def list_patients():
    try:
        # Lấy danh sách bệnh nhân từ Cassandra
        patients = Patient.objects.all()
        patients_list = list(patients)  # Chuyển thành list để dễ dàng xử lý
        
        # Tính toán số lượng bệnh nhân HGG và LGG
        hgg_count = 0
        lgg_count = 0
        age_sum = 0
        age_count = 0
        
        for patient in patients_list:
            if patient.grade == 'HGG':
                hgg_count += 1
            elif patient.grade == 'LGG':
                lgg_count += 1
                
            if patient.age:
                age_sum += patient.age
                age_count += 1
                
        avg_age = age_sum / age_count if age_count > 0 else None
        
        return render_template('patients.html', 
                              patients=patients_list, 
                              hgg_count=hgg_count, 
                              lgg_count=lgg_count,
                              avg_age=avg_age)
    except Exception as e:
        flash(f"Lỗi khi truy vấn dữ liệu bệnh nhân: {str(e)}", "danger")
        return redirect(url_for('index'))
@app.route('/statistics')
def statistics():
    # Thông báo cho người dùng biết tính năng đã bị vô hiệu hóa
    flash("Tính năng thống kê đã bị vô hiệu hóa trong phiên bản MapReduce", "info")
    # Chuyển hướng về trang chủ
    return redirect(url_for('index'))
@app.route('/survival-analysis')
def survival_analysis():
    # Thông báo cho người dùng biết tính năng đã bị vô hiệu hóa
    flash("Tính năng phân tích tiên lượng sống còn đã bị vô hiệu hóa trong phiên bản MapReduce", "info")
    # Chuyển hướng về trang chủ
    return redirect(url_for('index'))
@app.route('/report/<uuid:prediction_id>')
def generate_report(prediction_id):
    # Thông báo cho người dùng biết tính năng đã bị vô hiệu hóa
    flash("Tính năng tạo báo cáo đã bị vô hiệu hóa trong phiên bản MapReduce", "info")
    # Chuyển hướng về trang lịch sử dự đoán
    return redirect(url_for('prediction_history'))
@app.route('/compare', methods=['GET', 'POST'])
def compare_predictions():
    # Thông báo cho người dùng biết tính năng đã bị vô hiệu hóa
    flash("Tính năng so sánh dự đoán đã bị vô hiệu hóa trong phiên bản MapReduce", "info")
    # Chuyển hướng về trang lịch sử dự đoán
    return redirect(url_for('prediction_history'))
@app.route('/preview_slice', methods=['POST'])
def preview_slice():
    if 'mri_file' not in request.files:
        return jsonify({'error': 'Không có file nào được chọn'}), 400
        
    file = request.files['mri_file']
    slice_index = int(request.form.get('slice_index', 60))
    file_type = request.form.get('file_type', 'flair')
    
    if file.filename == '':
        return jsonify({'error': 'Không có file nào được chọn'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Định dạng file không được hỗ trợ'}), 400
    
    try:
        # Lưu file tạm thời
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_' + filename)
        file.save(temp_path)
        
        # Xử lý ảnh MRI
        img = nib.load(temp_path).get_fdata()
        img_slice = img[:, :, slice_index]
        
        # Chuyển đổi thành ảnh
        plt.figure(figsize=(6, 6))
        plt.imshow(img_slice, cmap='gray')
        plt.axis('off')
        
        # Lưu ảnh thành file tạm thời
        preview_filename = f"preview_{file_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        preview_path = os.path.join(app.config['RESULTS_FOLDER'], preview_filename)
        plt.savefig(preview_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Xóa file tạm thời
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'preview_url': url_for('static', filename=f'results/{preview_filename}')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# MAP: Patient search - search across different IDs
@app.route('/search', methods=['GET', 'POST'])
def search_patient():
    if request.method == 'POST':
        search_term = request.form.get('search_term', '')
        try:
            # MAP: Search in both Patient and PatientMapping tables
            patients = list(Patient.objects.filter(brats20id=search_term).allow_filtering())
            
            # If not found, search in other ID fields
            if not patients:
                # REDUCE: Find matching patient through different IDs
                mapping = PatientMapping.objects.filter(
                    brats17id=search_term
                ).allow_filtering().first() or PatientMapping.objects.filter(
                    brats18id=search_term
                ).allow_filtering().first() or PatientMapping.objects.filter(
                    brats19id=search_term
                ).allow_filtering().first() or PatientMapping.objects.filter(
                    tcga_id=search_term
                ).allow_filtering().first()
                
                if mapping:
                    patients = list(Patient.objects.filter(brats20id=mapping.brats20id).allow_filtering())
            
            return render_template('search_results.html', patients=patients, search_term=search_term)
        except Exception as e:
            flash(f"Error during search: {str(e)}", "danger")
            return redirect(url_for('index'))
    
    return render_template('search.html')

# REDUCE: View patient details by aggregating related data
@app.route('/patient/<brats_id>')
def view_patient(brats_id):
    try:
        # Get patient information from Cassandra
        patient = Patient.objects.get(brats20id=brats_id)
        
        # Get mapping information if available
        mapping = PatientMapping.objects.filter(brats20id=brats_id).first()
        
        # Get prediction history for this patient
        predictions = Prediction.objects.filter(brats_id=brats_id).allow_filtering()
        
        # Find patient's data directory
        patient_folder = os.path.join(BRATS_DATA_PATH, brats_id)
        has_mri_files = False
        max_slice = 155  # Default value
        
        if os.path.exists(patient_folder):
            # Check for necessary files
            flair_files = glob.glob(os.path.join(patient_folder, "*flair.nii*"))
            t1ce_files = glob.glob(os.path.join(patient_folder, "*t1ce.nii*"))
            seg_files = glob.glob(os.path.join(patient_folder, "*seg.nii*"))
            
            has_mri_files = len(flair_files) > 0 and len(t1ce_files) > 0 and len(seg_files) > 0
            
            # Get maximum slice number from first file
            if has_mri_files:
                try:
                    img = nib.load(flair_files[0]).get_fdata()
                    max_slice = img.shape[2] - 1
                except:
                    pass
        
        # REDUCE: Combine all data into a single view
        return render_template(
            'patient_detail.html',
            patient=patient,
            mapping=mapping,
            predictions=predictions,
            has_mri_files=has_mri_files,
            max_slice=max_slice
        )
    except Exception as e:
        flash(f"Error retrieving patient information: {str(e)}", "danger")
        return redirect(url_for('search_patient'))

# MAP & REDUCE: Prediction history - get all predictions and sort them
@app.route('/history')
def prediction_history():
    try:
        # MAP: Get all predictions from Cassandra
        predictions = Prediction.objects.all()
        
        # REDUCE: Sort by creation time (newest first)
        predictions_sorted = sorted(predictions, key=lambda p: p.created_at, reverse=True)
        
        # Convert timezone to Vietnam (UTC+7)
        vietnam_tz = timezone(timedelta(hours=7))
        for prediction in predictions_sorted:
            # Assume time is in UTC
            if prediction.created_at.tzinfo is None:
                utc_time = prediction.created_at.replace(tzinfo=timezone.utc)
            else:
                utc_time = prediction.created_at
            # Convert to Vietnam timezone
            prediction.created_at = utc_time.astimezone(vietnam_tz)
        
        return render_template('history.html', predictions=predictions_sorted)
    except Exception as e:
        flash(f"Error accessing prediction history: {str(e)}", "danger")
        return redirect(url_for('index'))

# REDUCE: View specific prediction result
@app.route('/prediction/<uuid:prediction_id>')
def view_prediction(prediction_id):
    try:
        prediction = Prediction.objects.get(id=prediction_id)
        result_path = os.path.join('results', prediction.result_path)
        
        # Get patient information if available
        patient = None
        if prediction.brats_id and prediction.brats_id != "unknown":
            patient = Patient.objects.filter(brats20id=prediction.brats_id).allow_filtering().first()
        
        return render_template( 
            'view_prediction.html', 
            title='Prediction Result', 
            prediction=prediction,
            patient=patient,
            image=result_path
        )
    except Exception as e:
        flash(f"Error viewing prediction: {str(e)}", "danger")
        return redirect(url_for('prediction_history'))

# MAP: Select data from BraTS
@app.route('/select-brats')
def select_brats_data():
    try:
        # MAP: Scan directory to find all patients
        patients = scan_brats_directory()
        return render_template('select_brats.html', patients=patients)
    except Exception as e:
        flash(f"Error scanning BraTS directory: {str(e)}", "danger")
        return redirect(url_for('index'))

# REDUCE: Process prediction from selected BraTS data
@app.route('/predict-brats', methods=['POST'])
def predict_brats():
    try:
        brats_id = request.form.get('brats_id')
        flair_path = request.form.get('flair_path')
        t1ce_path = request.form.get('t1ce_path')
        slice_index = int(request.form.get('slice_index', 60))
        notes = request.form.get('notes', '')
        
        if not all([brats_id, flair_path, t1ce_path]):
            flash("Missing required information!", "danger")
            return redirect(url_for('select_brats_data'))
        
        # REDUCE: Perform prediction (combines multiple inputs into single result)
        result_filename, prediction_id = perform_prediction(
            flair_path, t1ce_path, slice_index, brats_id, notes
        )
        
        # Redirect to result page
        return redirect(url_for('view_prediction', prediction_id=prediction_id))
    except Exception as e:
        flash(f"Error performing prediction: {str(e)}", "danger")
        return redirect(url_for('select_brats_data'))

# Handle MRI image upload and prediction
@app.route('/upload', methods=['GET'])
@app.route('/predict', methods=['GET', 'POST'])
def upload_and_predict():
    if not model_loaded:
        flash("Could not load model. Please check the path to the model file.", "danger")
        return render_template('upload.html')
        
    if request.method == 'POST':
        if 'flair_file' not in request.files or 't1ce_file' not in request.files:
            flash("Missing upload files!", "danger")
            return redirect(request.url)

        flair_file = request.files['flair_file']
        t1ce_file = request.files['t1ce_file']
        slice_index = int(request.form.get("slice_index", 60))
        brats_id = request.form.get("brats_id", "unknown")
        notes = request.form.get("notes", "")

        if flair_file.filename == '':
            flash("No FLAIR file selected!", "danger")
            return redirect(request.url)
        if t1ce_file.filename == '':
            flash("No T1CE file selected!", "danger")
            return redirect(request.url)
            
        if not allowed_file(flair_file.filename):
            flash("FLAIR file format not supported. Only .nii or .nii.gz files are accepted", "danger")
            return redirect(request.url)
        if not allowed_file(t1ce_file.filename):
            flash("T1CE file format not supported. Only .nii or .nii.gz files are accepted", "danger")
            return redirect(request.url)

        try:
            # MAP: Save uploaded files
            flair_filename = secure_filename(flair_file.filename)
            t1ce_filename = secure_filename(t1ce_file.filename)
            flair_path = os.path.join(app.config['UPLOAD_FOLDER'], flair_filename)
            t1ce_path = os.path.join(app.config['UPLOAD_FOLDER'], t1ce_filename)
            flair_file.save(flair_path)
            t1ce_file.save(t1ce_path)

            # REDUCE: Perform prediction and save result to database
            result_filename, prediction_id = perform_prediction(
                flair_path, t1ce_path, slice_index, brats_id, notes
            )
            
            # Redirect to result page
            return redirect(url_for('view_prediction', prediction_id=prediction_id))
        except Exception as e:
            flash(f"Error during processing: {str(e)}", "danger")
            return redirect(request.url)
    
    # GET request: display upload page with BraTS list
    try:
        # MAP: Get list of BraTS patients
        brats_patients = scan_brats_directory()
        
        # MAP: Get patient data from database to display grade
        patients_db = list(Patient.objects.all())
        
        return render_template('upload.html', brats_patients=brats_patients, patients_db=patients_db)
    except Exception as e:
        flash(f"Error scanning BraTS directory: {str(e)}", "warning")
        return render_template('upload.html', brats_patients=[], patients_db=[])

# Display slice from patient data (AJAX endpoint)
@app.route('/patient-slice-data', methods=['POST'])
def patient_slice_data():
    try:
        # Process request data
        print("Request data:", request.form)
        
        # Check if request is from BraTS modal
        if 'file_path' in request.form:
            # Case from BraTS selection modal
            brats_id = request.form.get('brats_id', 'unknown')
            file_type = request.form.get('file_type', '')
            file_path = request.form.get('file_path')
            slice_index = int(request.form.get('slice_index', 60))
            
            print(f"Processing BraTS modal request - File path: {file_path}")
            
            # Use file path from request
            mri_path = file_path
            
            # Check if file exists
            if not os.path.exists(mri_path) or not os.path.isfile(mri_path):
                print(f"File does not exist: {mri_path}")
                return jsonify({'error': f'File not found: {mri_path}'}), 404
                
        elif 'brats_id' in request.form:
            # Case from patient detail view
            brats_id = request.form.get('brats_id')
            file_type = request.form.get('file_type', '')
            slice_index = int(request.form.get('slice_index', 60))
            
            print(f"Processing patient detail request - BraTS ID: {brats_id}, File type: {file_type}")
            
            # Find patient folder
            patient_folder = os.path.join(BRATS_DATA_PATH, brats_id)
            print(f"Looking in folder: {patient_folder}")
            
            if not os.path.exists(patient_folder):
                print(f"Patient folder not found: {patient_folder}")
                return jsonify({'error': 'Patient folder not found'}), 404
            
            # Find corresponding MRI file
            file_pattern = f"*{file_type}.nii*"
            mri_files = glob.glob(os.path.join(patient_folder, file_pattern))
            print(f"Found files matching pattern '{file_pattern}': {mri_files}")
            
            if not mri_files:
                print(f"No matching files found for {file_type}")
                return jsonify({'error': f'File {file_type} not found'}), 404
            
            # Get first file
            mri_path = mri_files[0]
            
            # Check if file exists
            if not os.path.isfile(mri_path):
                print(f"File is not a valid file: {mri_path}")
                return jsonify({'error': f'Invalid file: {mri_path}'}), 404
        else:
            print("Neither 'file_path' nor 'brats_id' found in form data")
            return jsonify({'error': 'Missing patient information or file path'}), 400
            
        # MAP: Process MRI image
        try:
            print(f"Loading NIfTI file: {mri_path}")
            img = nib.load(mri_path).get_fdata()
            
            # Check image dimensions and slice index
            if slice_index < 0 or slice_index >= img.shape[2]:
                print(f"Invalid slice index: {slice_index}, image shape: {img.shape}")
                return jsonify({
                    'error': f'Invalid slice index. Value must be between 0 and {img.shape[2]-1}.'
                }), 400
                
            img_slice = img[:, :, slice_index]
            print(f"Slice extracted successfully, shape: {img_slice.shape}")
        except Exception as e:
            print(f"Error loading NIfTI file: {str(e)}")
            return jsonify({'error': f'Error reading NIfTI file: {str(e)}'}), 500
        
        # Create image with appropriate colormap
        try:
            print(f"Creating image with colormap: {file_type}")
            if file_type == 'seg':
                cmap = create_tumor_segmentation_cmap()
                vmin, vmax = 0, 3
            else:
                cmap = 'gray'
                vmin, vmax = None, None
            
            # Normalize data to avoid display issues
            if img_slice.max() > 0:  # Avoid division by zero
                img_slice_normalized = img_slice / img_slice.max()
            else:
                img_slice_normalized = img_slice
            
            # Convert to image
            plt.figure(figsize=(6, 6))
            plt.imshow(img_slice_normalized, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.axis('off')
            
            # Add colorbar for segmentation
            if file_type == 'seg':
                plt.colorbar(ticks=[0, 1, 2, 3]).set_ticklabels(['Background', 'NCR/NET', 'ED', 'ET'])
        except Exception as e:
            print(f"Error creating image: {str(e)}")
            return jsonify({'error': f'Error creating image: {str(e)}'}), 500
        
        # Convert image to base64 string
        try:
            print("Converting image to base64")
            from io import BytesIO
            import base64
            buffer = BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            buffer.seek(0)
            
            # Encode to base64
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            print("Image successfully converted to base64")
        except Exception as e:
            print(f"Error converting image to base64: {str(e)}")
            return jsonify({'error': f'Error converting image: {str(e)}'}), 500
        
        # Return base64 data
        return jsonify({
            'success': True,
            'image_data': f'data:image/png;base64,{img_base64}'
        })
    except Exception as e:
        print(f"Unexpected error in patient_slice_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)