# manage.py
import os
from werkzeug import secure_filename
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from result import recognition


app = Flask(__name__)
app.config['SECRET_KEY'] = '44617457d542163d10ada66726b31ef80a88ac1a41013ea5'
bootstrap = Bootstrap(app)
app.config['UPLOADED_PHOTO_DEST'] = os.path.dirname(os.path.abspath(__file__))+'/images'


class FileForm(FlaskForm):
    # 定义表单类
    photo = FileField('请上传验证码图片', validators=[
        FileAllowed(['jpg', 'png'],  u'只能上传图片'),
        FileRequired(u'文件未选择')])
    submit = SubmitField('提交并查看返回结果')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = FileForm()
    if form.validate_on_submit():
        filename = secure_filename(form.photo.data.filename)
        form.photo.data.save(os.path.join(app.config['UPLOADED_PHOTO_DEST'], filename))
        label = recognition(filename)
        with open("pre_label_list.txt", "a+") as f:
            f.write('%s' % label[0] + '\n')
        file_url = filename
    else:
        file_url = None
        label = [None]
    return render_template('index.html', form=form, file_url=file_url, label=label[0])


@app.route('/uploads/<filename>')
def up_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOADED_PHOTO_DEST'], filename)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500