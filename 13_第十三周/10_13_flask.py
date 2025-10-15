from flask import Flask, request, redirect, url_for, jsonify
from flask import render_template

app = Flask(__name__)

import json

import pymysql
conn = pymysql.connect(host='localhost', user='root', password='123456', database='my_db', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
cursor = conn.cursor()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/demo')
def demo():
    print("Accessed /demo")
    return render_template('demo.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        # 处理 POST 请求（表单提交）
        username = request.form.get("username")
        password = request.form.get("password")
        print(f"Login attempt: {username}, {password}")
        
        if username == "admin" and password == "123456":
            return redirect(url_for('welcome'))
        else:
            error_mag = "用户名或密码错误"
            return render_template('login.html', error_mag=error_mag)
    else:
        # 处理 GET 请求（显示登录页面）
        return render_template('login.html')

@app.route('/welcome')
def welcome():
    sql = "SELECT * FROM exam"
    cursor.execute(sql)
    data = cursor.fetchall()
    print("查询到的数据：", data)
    print("数据行数：", len(data))
    if data:
        print("第一行的键：", data[0].keys())
    return render_template('welcome.html', userlist=data)

@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == "POST":
        try:
            ExamNo = request.form.get("ExamNo")
            stuNo = request.form.get("stuNo")
            writtenExam = request.form.get("writtenExam")
            LabExam = request.form.get("LabExam")
            sql = "INSERT INTO exam (ExamNo, stuNo, writtenExam, LabExam) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql, (ExamNo, stuNo, writtenExam, LabExam))
            conn.commit()
            print(f"Added exam: {ExamNo}, {stuNo}, {writtenExam}, {LabExam}")
            # 处理添加逻辑
            return redirect(url_for('welcome'))
        except Exception as e:
            print(f"Error adding exam: {e}")
            error_mag = "添加失败，请检查输入"
            return render_template('add.html', error_mag=error_mag)
    else:
        return render_template('add.html')

@app.route('/add_ajax', methods=['GET', 'POST'])
def add_ajax():
    if request.method == "POST":
        try:
            # 获取前端提交的数据
            data = json.loads(request.get_data())
            ExamNo = data.get('ExamNo')
            stuNo = data.get('stuNo')
            writtenExam = data.get('writtenExam')
            LabExam = data.get('LabExam')
            print(f"接收的数据：{data}")
            sql = "INSERT INTO exam (ExamNo, stuNo, writtenExam, LabExam) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql, (ExamNo, stuNo, writtenExam, LabExam))
            conn.commit()
            print("添加成功！")
            return jsonify({
                'success': True,
                'msg': '数据提交成功',
                'redirect_url': url_for('welcome')
            })
        except Exception as e:
            print(f"添加失败！错误信息：{e}")
            conn.rollback()  # 回滚事务
            return jsonify({
                'success': False,
                'msg': f"添加失败！{str(e)}"
            }), 400
        
    else:
        return render_template('add_ajax.html')

@app.route('/delete', methods=['GET', 'POST'])
def delete():
    if request.method == "POST":
        try:
            choice = request.form.get("choice")
            sql = "DELETE FROM exam WHERE id = %s"
            cursor.execute(sql, (choice,))
            conn.commit()
            print(f"Deleted exam with id: {choice}")
            return redirect(url_for('welcome'))
        except Exception as e:
            print(f"Error deleting exam: {e}")
            error_mag = "删除失败，请检查输入"
            return render_template('delete.html', error_mag=error_mag)
    else:
        return render_template('delete.html')

@app.route('/delete_ajax', methods=['GET', 'POST'])
def delete_ajax():
    if request.method == 'POST':
        try:
            # 获取前端发送的 JSON 数据
            data = request.get_json()
            id = data.get('choice')  # 提取 choice 字段的值
            print(f"将要删除的记录的id：{id}")

            # 修复 SQL 语法：使用 WHERE 而不是 with，使用 ExamNo 字段
            sql = "DELETE FROM exam WHERE id = %s"
            cursor.execute(sql, (id,))
            conn.commit()
            print(f"成功删除id={id}的记录！")

            return jsonify({
                'success': True,
                'msg': '删除成功',
                'redirect_url': url_for('welcome')
            })
        except Exception as e:
            print(f"删除失败！错误信息：{e}")
            conn.rollback()  # 回滚事务
            return jsonify({
                'success': False,
                'msg': f"删除失败！{str(e)}"
            }), 400
    else:
        return render_template('delete_ajax.html')

@app.route('/edit/<exam_id>', methods=['GET', 'POST'])
def edit(exam_id):
    if request.method == 'POST':
        # 【POST 逻辑】：接收表单修改后的数据，更新数据库
        try:

            new_ExamNo = request.form.get("ExamNo")       # 新考试编号
            new_stuNo = request.form.get("stuNo")       # 新学号
            new_writtenExam = request.form.get("writtenExam") # 新笔试成绩
            new_LabExam = request.form.get("LabExam")     # 新实验成绩
            # 执行 SQL 更新语句
            sql = "UPDATE exam SET  ExamNo = %s, stuNo = %s, writtenExam = %s, LabExam = %s WHERE id = %s"
            cursor.execute(sql, (new_ExamNo, new_stuNo, new_writtenExam, new_LabExam, exam_id))
            conn.commit()  # 提交事务
            return redirect(url_for('welcome'))  # 更新后跳回列表页
        except Exception as e:
            print(f"更新数据出错：{e}")
            error_mag = "更新失败，请检查输入！"
            # 若更新失败，查询原数据回显到编辑页
            sql = "SELECT * FROM exam WHERE id = %s"
            cursor.execute(sql, (exam_id,))
            exam_data = cursor.fetchone()
            return render_template('edit.html', exam_data=exam_data, error_mag=error_mag)
    else:
        # 【GET 逻辑】：根据 id 查询待编辑的记录，渲染到编辑页
        sql = "SELECT * FROM exam WHERE id = %s"
        cursor.execute(sql, (exam_id,))
        exam_data = cursor.fetchone()
        if not exam_data:  # 若记录不存在
            return "记录不存在", 404
        return render_template('edit.html', exam_data=exam_data)

@app.route('/delete01/<exam_id>', methods=['GET', 'POST'])
def delete01(exam_id):
    if request.method == 'POST':    
        try:
            sql = "DELETE FROM exam WHERE id = %s"
            cursor.execute(sql, (exam_id,))
            conn.commit()
            print(f"Deleted exam with id: {exam_id}")
            return redirect(url_for('welcome'))
        except Exception as e:
            print(f"Error deleting exam: {e}")
            error_mag = "删除失败，请检查输入"
            return render_template('delete.html', error_mag=error_mag)
    else:
        # 【GET 逻辑】：根据 id 查询待删除的记录，渲染到删除确认页
        sql = "SELECT * FROM exam WHERE id = %s"
        cursor.execute(sql, (exam_id,))
        exam_data = cursor.fetchone()
        if not exam_data:  # 若记录不存在
            return "记录不存在", 404
        return render_template('delete_confirm.html', exam_data=exam_data)

if __name__ == '__main__':
    app.run(debug=True)