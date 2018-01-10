# manage.py
import os
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from result import recognition


app = Flask(__name__)
app.config['SECRET_KEY'] = '44617457d542163d10ada66726b31ef80a88ac1a41013ea5'
bootstrap = Bootstrap(app)
app.config['UPLOADED_PHOTO_DEST'] = os.path.dirname(os.path.abspath(__file__))+'/images'
app.config['UPLOADED_PHOTO_ALLOW'] = IMAGES
photos = UploadSet('PHOTO')
configure_uploads(app, photos)
patch_request_class(app)


class FileForm(FlaskForm):
    # 定义表单类
    photo = FileField('请上传验证码图片', validators=[
        FileAllowed(photos, u'只能上传图片'),
        FileRequired(u'文件未选择')])
    submit = SubmitField('提交并查看返回结果')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = FileForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)
        recognition()
        with open("pre_l.txt", "r") as f:
            label = f.readline()
        os.remove("pre_l.txt")
        with open("pre_label_list.txt", "a+") as f_w:
            f_w.write('%s' % label)
        filepath = 'images/'
        for name in os.listdir(filepath):
            filename1 = filepath + name
            os.remove(filename1)
    else:
        file_url = None
        label = None
    return render_template('index.html', form=form, file_url=file_url, label=label)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

