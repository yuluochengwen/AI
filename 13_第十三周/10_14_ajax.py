from flask import Flask, request, redirect, url_for, jsonify
from flask import render_template
import json
app = Flask(__name__)

import pymysql
conn = pymysql.connect(host='localhost', user='root', password='123456', database='my_db', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
cursor = conn.cursor()

@app.route('/sendjson1',methods=['GET','POST'])
def sendjson1():
    msg = {}
    msg['name'] = 'tom'
    msg['time'] = '2016'
    print(msg)
    return jsonify(msg)

@app.route("/ajax01", methods=['GET','POST'])
def ajax01():
    return render_template('ajax01.html')

@app.route('/sendjson2', methods=['GET', 'POST'])
def sendjson2():
    try:
        data = json.loads(request.get_data())   # request_data()用来获取前端请求发送的数据
        name = data.get('name', '')
        age = data.get('age', 0)
        location = data.get('location', '')
        data['time'] = '2025-10-14'
        print(f"接收到的数据: {data}")
        return jsonify(data)
    except Exception as e:
        print(f"错误: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/ajax02', methods=['get', 'post'])
def ajax02():
    return render_template('ajax02.html')

@app.route('/sendjson3', methods=['GET', 'POST'])
def sendjson3():
    data = request.get_data()
    print(request.form['start'])
    print(request.form['end'])

    msg = {}
    msg['name'] = 'tom'
    msg['time'] = '2016'
    msg['location'] = '苏州'
    print(msg)
    return jsonify(msg)

@app.route('/ajax03', methods=['GET', 'POST'])
def ajax03():
    return render_template('ajax03.html')







if __name__ == '__main__':
    app.run(debug=True)