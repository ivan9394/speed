from flask import Flask, request, jsonify,render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import os


app = Flask(__name__)
app.config['UPLOADS_DEFAULT_DEST'] = os.getcwd() + "/upload/vedio"

vedios = UploadSet('vedios')
configure_uploads(app, vedios)

patch_request_class(app)  # 文件大小限制，默认为16MB
@app.route('/', methods = ['GET','POST'])
def up_loadfiles():
    if request.method == 'POST' and 'vedio' in request.files:
        filename = vedios.save(request.files['vedio'])
        file_url = vedios.url(filename)
        return render_template('index.html')
        # return html + '<br><img src=' + file_url + '>'
    return render_template('index.html')
        

@app.route('/upload')
def upload_video():
    # Get the file from the request
    return render_template('upload_vedio.html')

if __name__ == '__main__':
    app.run()