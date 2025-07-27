from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import matplotlib.colors as mcolors
from datetime import datetime

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Cấu hình cơ sở dữ liệu
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Cấu hình Login Manager
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Vui lòng đăng nhập để truy cập trang này.'
login_manager.login_message_category = 'info'

# Định nghĩa thư mục lưu trữ
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
RESULTS_FOLDER = os.path.join(STATIC_FOLDER, 'results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'nii'}

# Tạo thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'img'), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'css'), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'js'), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'lib'), exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Định nghĩa kích thước ảnh đầu vào
IMG_SIZE = 128

# Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    slice_index = db.Column(db.Integer, nullable=False)
    result_path = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Prediction('{self.filename}', '{self.slice_index}', '{self.created_at}')"

# Forms
class RegistrationForm(FlaskForm):
    username = StringField('Tên người dùng', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Mật khẩu', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Xác nhận mật khẩu', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Đăng ký')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Tên người dùng đã tồn tại. Vui lòng chọn tên khác.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email đã được sử dụng. Vui lòng sử dụng email khác.')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Mật khẩu', validators=[DataRequired()])
    remember = BooleanField('Ghi nhớ đăng nhập')
    submit = SubmitField('Đăng nhập')

# Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Hàm tính Dice Coefficient cho mô hình
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4  # Số lớp phân đoạn (background + tumor parts)
    total_loss = 0
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:, :, :, i])
        y_pred_f = K.flatten(y_pred[:, :, :, i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        total_loss = total_loss + loss if i > 0 else loss
    return total_loss / class_num

# Load mô hình đã huấn luyện
try:
    model = tf.keras.models.load_model(
        "model/3D_MRI_Brain_tumor_segmentation.h5",
        custom_objects={'dice_coef': dice_coef},
        compile=False
    )
    model_loaded = True
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    model_loaded = False

# Kiểm tra định dạng file hợp lệ
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Tiền xử lý ảnh MRI từ file .nii
def preprocess_image(file_path, slice_index=60):
    img = nib.load(file_path).get_fdata()
    img = img[:, :, slice_index]  # Lấy lát cắt
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize về (128, 128)
    
    # Tránh lỗi chia cho 0 khi chuẩn hóa
    img_max = np.max(img_resized)
    img_normalized = img_resized / img_max if img_max > 0 else img_resized
    
    return img_normalized

# Tạo colormap tùy chỉnh tương ứng với bảng phân đoạn
def create_tumor_segmentation_cmap():
    # Màu cho từng lớp:
    # 0: Background (đen)
    # 1: NCR/NET (đỏ)
    # 2: ED (vàng)
    # 3: ET (xanh)
    colors = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1)]
    return mcolors.ListedColormap(colors)

# Hàm thực hiện dự đoán khối u trên ảnh MRI
def perform_prediction(flair_path, t1ce_path, slice_index, user_id=None):
    # Tiền xử lý ảnh FLAIR và T1CE
    X_flair = preprocess_image(flair_path, slice_index)
    X_t1ce = preprocess_image(t1ce_path, slice_index)
    
    # Ghép thành tensor có 2 kênh
    X = np.stack([X_flair, X_t1ce], axis=-1)  # (128, 128, 2)
    
    # Định dạng đầu vào đúng chuẩn `(1, 128, 128, 2)`
    X_input = np.expand_dims(X, axis=0)
    
    # Dự đoán
    pred = model.predict(X_input)
    
    # Lấy lớp có giá trị lớn nhất
    prediction = np.argmax(pred[0], axis=-1)

    # Tạo colormap tùy chỉnh
    tumor_cmap = create_tumor_segmentation_cmap()

    # Hiển thị kết quả với colormap tùy chỉnh
    plt.figure(figsize=(14, 7))
    
    # Ảnh MRI FLAIR gốc
    plt.subplot(1, 2, 1)
    plt.imshow(X_flair, cmap='gray')
    plt.title(f"Lát cắt {slice_index} - Ảnh MRI (FLAIR)", fontsize=14)
    plt.axis("off")
    
    # Ảnh phân đoạn
    plt.subplot(1, 2, 2)
    im = plt.imshow(prediction, cmap=tumor_cmap, vmin=0, vmax=3)
    plt.title(f"Phân đoạn tại lát cắt {slice_index}", fontsize=14)
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['Background', 'NCR/NET', 'ED', 'ET'])
    plt.axis("off")

    # Tạo tên file kết quả duy nhất dựa trên timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    result_filename = f"result_{timestamp}.png"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    
    # Lưu kết quả với chất lượng cao
    plt.tight_layout()
    plt.savefig(result_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Lưu thông tin dự đoán vào cơ sở dữ liệu nếu người dùng đã đăng nhập
    if user_id:
        filename = os.path.basename(flair_path)
        prediction_record = Prediction(
            filename=filename,
            slice_index=slice_index,
            result_path=result_filename,  # Lưu tên file thay vì đường dẫn đầy đủ
            user_id=user_id
        )
        db.session.add(prediction_record)
        db.session.commit()
    
    return result_filename  # Trả về tên file kết quả

# Khởi tạo cơ sở dữ liệu khi ứng dụng khởi chạy (sửa lỗi)
with app.app_context():
    db.create_all()

# Routes
# Trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Đăng ký
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        user = User(
            username=form.username.data,
            email=form.email.data,
            password=hashed_password
        )
        db.session.add(user)
        db.session.commit()
        flash(f'Tài khoản đã được tạo cho {form.username.data}! Bạn có thể đăng nhập ngay bây giờ.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', title='Đăng Ký', form=form)

# Đăng nhập
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            flash(f'Đăng nhập thành công! Xin chào {user.username}.', 'success')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Đăng nhập không thành công. Vui lòng kiểm tra email và mật khẩu.', 'danger')
    
    return render_template('login.html', title='Đăng Nhập', form=form)

# Đăng xuất
@app.route('/logout')
def logout():
    logout_user()
    flash('Bạn đã đăng xuất thành công.', 'success')
    return redirect(url_for('index'))

# Trang hồ sơ người dùng
@app.route('/profile')
@login_required
def profile():
    # Lấy lịch sử dự đoán của người dùng, sắp xếp theo thời gian mới nhất
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template('profile.html', title='Hồ Sơ', predictions=predictions)

# Xem kết quả dự đoán cụ thể
@app.route('/prediction/<int:prediction_id>')
@login_required
def view_prediction(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    # Kiểm tra xem người dùng hiện tại có quyền xem dự đoán này không
    if prediction.user_id != current_user.id:
        flash('Bạn không có quyền xem kết quả dự đoán này.', 'danger')
        return redirect(url_for('profile'))
    
    result_path = os.path.join('results', prediction.result_path)
    return render_template( 
        'view_prediction.html', 
        title='Kết Quả Dự Đoán', 
        prediction=prediction,
        image=result_path
    )

# Xử lý tải ảnh MRI và dự đoán
@app.route('/predict', methods=['GET', 'POST'])
@login_required  # Yêu cầu đăng nhập để sử dụng tính năng dự đoán
def upload_and_predict():
    if not model_loaded:
        flash("Không thể tải mô hình. Vui lòng kiểm tra đường dẫn đến file mô hình.", "danger")
        return render_template('upload.html')
        
    if request.method == 'POST':
        if 'flair_file' not in request.files or 't1ce_file' not in request.files:
            flash("Thiếu file upload!", "danger")
            return redirect(request.url)

        flair_file = request.files['flair_file']
        t1ce_file = request.files['t1ce_file']
        slice_index = int(request.form.get("slice_index", 60))

        if flair_file.filename == '':
            flash("Chưa chọn file FLAIR!", "danger")
            return redirect(request.url)
        if t1ce_file.filename == '':
            flash("Chưa chọn file T1CE!", "danger")
            return redirect(request.url)
            
        if not allowed_file(flair_file.filename):
            flash("File FLAIR không đúng định dạng. Chỉ chấp nhận file .nii", "danger")
            return redirect(request.url)
        if not allowed_file(t1ce_file.filename):
            flash("File T1CE không đúng định dạng. Chỉ chấp nhận file .nii", "danger")
            return redirect(request.url)

        try:
            flair_filename = secure_filename(flair_file.filename)
            t1ce_filename = secure_filename(t1ce_file.filename)
            flair_path = os.path.join(app.config['UPLOAD_FOLDER'], flair_filename)
            t1ce_path = os.path.join(app.config['UPLOAD_FOLDER'], t1ce_filename)
            flair_file.save(flair_path)
            t1ce_file.save(t1ce_path)

            # Thực hiện dự đoán và lưu kết quả vào cơ sở dữ liệu
            result_filename = perform_prediction(flair_path, t1ce_path, slice_index, current_user.id)
            result_path = os.path.join('results', result_filename)
            
            return render_template('result.html', image=result_path, slice_index=slice_index)
        except Exception as e:
            flash(f"Lỗi xảy ra trong quá trình xử lý: {str(e)}", "danger")
            return redirect(request.url)
    
    return render_template('upload.html')

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)