"""Thêm cột role vào bảng user

Revision ID: 64850960e259
Revises: 
Create Date: 2025-03-23 21:53:52.177418

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision = '64850960e259'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    columns = [col['name'] for col in inspector.get_columns('user')]
    if 'role' not in columns:
        with op.batch_alter_table('user', schema=None) as batch_op:
            batch_op.add_column(sa.Column('role', sa.String(length=20), nullable=True))
    else:
        print("Column 'role' đã tồn tại, bỏ qua thêm cột.")


def downgrade():
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    columns = [col['name'] for col in inspector.get_columns('user')]
    if 'role' in columns:
        with op.batch_alter_table('user', schema=None) as batch_op:
            batch_op.drop_column('role')
    else:
        print("Column 'role' không tồn tại, bỏ qua xóa cột.")
