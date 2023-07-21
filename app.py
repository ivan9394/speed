import os
import sys
import speed
import datetime
import threading
import uuid
from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from speed import speed_cal, process_frame, get_landmark
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf import FlaskForm 
from wtforms import StringField, PasswordField, SubmitField, HiddenField,FileField,SelectField
from wtforms.validators import DataRequired, EqualTo, Email
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect

from werkzeug.security import generate_password_hash, check_password_hash

UPLOAD_FOLDER = os.getcwd() + "/upload/video"
ALLOWED_EXTENSIONS = {'mp4','mov','avi', 'wmv', 'flv', 'mkv', 'm4v'}
# SQLite URI compatible
WIN = sys.platform.startswith('win')
if WIN:
    prefix = 'sqlite:///'
else:
    prefix = 'sqlite:////'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# getenv获取环境变量中的数据库存储地址, data.db是数据库名称, 可以存储多张数据表
print(app.root_path)
app.config['SQLALCHEMY_DATABASE_URI'] = prefix + os.path.join(app.root_path, 'data.db')
print(app.config['SQLALCHEMY_DATABASE_URI'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.static_url_path = '/static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.secret_key = 'ivan9394'
CSRFProtect(app)
db = SQLAlchemy(app)


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    __tablename__ = 'userinfo'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    date_created = db.Column(db.DateTime, default=datetime.datetime.now())
    body_height = db.Column(db.Float)
    ankle2shoulder = db.Column(db.Float)
    body_weight = db.Column(db.Integer)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class User_Speed(db.Model):
    __tablename__ = 'user_speed_list'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index = True)
    act_type = db.Column(db.String(16))
    max_speed = db.Column(db.Float)
    max_height = db.Column(db.Integer)
    max_power = db.Column(db.Integer)
    filename = db.Column(db.String(64))
    upload_time = db.Column(db.DateTime, default=datetime.datetime.now())
    body_height = db.Column(db.Float)
    ankle2shoulder = db.Column(db.Float)
    body_weight = db.Column(db.Float)
    load_weight = db.Column(db.Float)



@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('用户名', validators=[DataRequired("请输入昵称")])
    password = PasswordField('密码', validators=[DataRequired("请输入密码")])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('邮箱', validators=[DataRequired("请输入邮箱"), Email("请输入邮箱")])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    body_height = StringField('身高/米', validators=[DataRequired("请输入身高")])
    ankle2shoulder = StringField('脚跟至肩关节高度/米', default='0')
    body_weight = StringField('体重/kg',validators = [DataRequired("请输入体重")])
    submit = SubmitField('Register')

class UploadForm(FlaskForm):
    file = FileField('视频', validators=[DataRequired("请上传视频")])
    act_type = SelectField('跳跃类型', validators=[DataRequired("请选择跳跃类型")],choices=['反向跳', '无反向跳', '跳深','助跑跳','快速跳深','负重蹲跳'])
    load_weight = StringField('负重/kg')
    submit = SubmitField('上传')

@app.route('/')
def index():
    if current_user.is_authenticated:
        current_list = User_Speed.query.filter_by(username = current_user.username).order_by(User_Speed.id.desc()).all()
        return render_template('speed_list.html', speed_list = current_list)
    else:
        return render_template('login.html', form = LoginForm())

@app.route('/login', methods=['GET', 'POST'])
def login():
    form1 = LoginForm()
    if request.method == 'GET':
        return render_template('login.html', form = form1)
    elif request.method == 'POST':
        if form1.validate_on_submit():
            user = User.query.filter_by(username=form1.username.data).first()
            if user:
                if user.check_password(form1.password.data):
                    login_user(user)
                    next_page = request.args.get('next')
                    current_list = User_Speed.query.filter_by(username = current_user.username).order_by(User_Speed.id.desc()).all()
                    return render_template('speed_list.html', speed_list = current_list)
            flash('Invalid username or password')
            return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form1 = RegisterForm()
    if request.method == 'GET':
        return render_template('register.html', form = form1)
    elif request.method == 'POST':
        if form1.validate_on_submit():
            user = User(username=form1.username.data
                        ,email=form1.email.data
                        ,body_height=float(form1.body_height.data)
                        ,ankle2shoulder=float(form1.ankle2shoulder.data)
                        ,body_weight=float(form1.body_weight.data)
                        )
            user.set_password(form1.password.data)
            db.session.add(user)
            db.session.commit()
            flash('谢谢支持,您已成功注册!')
            return redirect(url_for('login'))
        return render_template('register.html', form=form1)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@login_required
@app.route('/speed_list')
def speed_list():
    return render_template('speed_list.html', name = current_user.username)

@login_required
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    speed_form = UploadForm()
    if request.method == 'POST':
        if speed_form.validate_on_submit():
            if allowed_file(speed_form.file.data.filename):
                filename = secure_filename(speed_form.file.data.filename)
                new_filename = current_user.username + '_' + str(uuid.uuid4()) + filename
                request.files['file'].save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
                max_speed = speed_cal(ref_height = current_user.body_height, input_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename), method = 'cmj', height_type = 'bodyheight')
                max_height = round((0.5 * max_speed * max_speed / 9.8)*100)
                try:
                    load_weight = float(speed_form.load_weight.data)
                except:
                    load_weight = 0
                max_power = round(max_speed * (current_user.body_weight + load_weight) * 9.8)
                max_speed = round(max_speed,2)
                s_list = User_Speed(username =current_user.username
                        ,filename = new_filename
                        ,body_height = current_user.body_height
                        ,body_weight = current_user.body_weight
                        ,act_type = speed_form.act_type.data
                        ,max_speed = max_speed
                        ,max_height = max_height
                        ,max_power = max_power
                        ,load_weight = load_weight)
                db.session.add(s_list)
                db.session.commit()
                current_list = User_Speed.query.filter_by(username = current_user.username).order_by(User_Speed.id.desc()).all()
                return render_template('speed_list.html', speed_list = current_list)
            else:
                flash('视频名称'+speed_form.file.data + '不符合要求')
                return render_template('upload_video.html', form = speed_form)
        else:
            flash('输入格式不正确')
            return render_template('upload_video.html', form = speed_form)
    elif request.method == 'GET':
        return render_template('upload_video.html', form = speed_form)

@app.route('/delete/<int:id>')
def my_delete(id):
    s_list = User_Speed.query.get(id)
    del_file = os.path.join(app.config['UPLOAD_FOLDER'], s_list.filename)
    if del_file:
        if os.path.exists(del_file):
            os.remove(del_file)  # 删除文件
            flash('文件删除成功')
        else:
            flash('文件不存在')
    else:
        flash('未提供要删除的文件名')
    db.session.delete(s_list)
    db.session.commit()
    current_list = User_Speed.query.filter_by(username = current_user.username).order_by(User_Speed.id.desc()).all()
    return render_template('speed_list.html', speed_list = current_list)


def init_db():
    with app.app_context():
        db.create_all()
        print('Database initialized successfully.')


#注释以下代码
if __name__ == '__main__':
    init_db()
    app.run(debug=True)