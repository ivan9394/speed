import os
import speed
import uuid
from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from speed import speed_cal, process_frame, get_landmark

UPLOAD_FOLDER = os.getcwd() + "/upload/video"
ALLOWED_EXTENSIONS = {'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.secret_key = 'ivan9394'

# patch_request_class(app)  # 文件大小限制，默认为16MB
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        print(request.files)
        if 'file' not in request.files:
            print('请选择视频')
            flash('请选择视频')
            return redirect(url_for('upload_file'))
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            print('找不到该文件')
            flash('找不到该文件')
            return redirect(url_for('upload_file'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(f'文件名称 {filename}')
            new_filename = str(uuid.uuid4()) + filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
            max_speed = speed_cal(ref_height = request.form.get(key = 'user_bodyheight', type = float), input_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename), method = 'cmj', height_type = 'bodyheight')
            print(f'最大速度: {max_speed} 米/秒')
            height = round((0.5 * max_speed * max_speed / 9.8)*100)
            max_power = max_speed * (request.form.get(key = 'user_weight', type = float) + request.form.get(key = 'upload_weight', type = int)) * 9.8
            return render_template('result_table.html', max_speed = round(max_speed,2), max_height = height, max_power = round(max_power))
    else:
        return render_template('index.html')


        

@app.route('/upload')
def upload_video():
    # Get the file from the request
    return render_template('upload_video2.html')

if __name__ == '__main__':
    app.run(host='47.99.145.151', port= 80)