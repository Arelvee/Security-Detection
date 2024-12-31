from flask import Flask, request, redirect, url_for, render_template,flash, session

import sqlite3
import base64
#from flask_socketio import SocketIO, emit

app = Flask(__name__)

app.secret_key = '123456'




@app.route('/')
def home():
    return render_template('login.html')

@app.route('/logout')
def logout():
    return render_template('login.html')

@app.route('/back')
def back():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    conn = sqlite3.connect('mydb.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
    user = c.fetchone()
    conn.close()
    if user:
        session['username'] = username  # Store username in session
        return render_template("index.html")
    else:
        return redirect(url_for('home', message='Login failed. Invalid username or password.'))
@app.route('/join', methods=['GET', 'POST'])
def join():
    if request.method == 'POST':
        Fname = request.form['Fname']
        Lname = request.form['Lname']
        color = request.form['color']
        age = request.form['age']
        gender = request.form['gender']
        PhoneNo = request.form['PhoneNo']

        with sqlite3.connect("mydb.db") as users:
            cursor = users.cursor()
            cursor.execute("INSERT INTO car_user \
                           (Fname,Lname,color,age,gender,PhoneNo) VALUES (?,?,?,?,?,?)",
                           (Fname,Lname,color,age,gender,PhoneNo))
            users.commit()
        return render_template("index.html")
    else:

        return render_template('join.html')

@app.route('/participants', methods=['GET'])
def participants():
    search_query = request.args.get('search', '')
    filter_by = request.args.get('filter', '')

    conn = sqlite3.connect('mydb.db')  # Connect to the correct database
    cursor = conn.cursor()

    if search_query and filter_by:
        # Constructing the SQL query dynamically based on the selected filter
        query = f"SELECT * FROM detected_frames WHERE {filter_by} LIKE ?"
        cursor.execute(query, ('%' + search_query + '%',))
    else:
        cursor.execute('SELECT * FROM detected_frames')

    data = cursor.fetchall()
    conn.close()

    # Convert image data to base64
    data_with_base64_images = []
    for participant in data:
        image_data = participant[2]  # Assuming image data is stored in the 4th column
        if image_data:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            participant_with_image = list(participant)
            participant_with_image[2] = image_base64
            data_with_base64_images.append(participant_with_image)
        else:
            data_with_base64_images.append(participant)


    return render_template("participants.html", data=data_with_base64_images)

@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if request.method == 'POST':
        username = session['username']  # Assuming you store the username in session after login

        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        # Verify if new password matches the confirm password
        if new_password != confirm_password:
            flash('New password and confirm password do not match!', 'error')
            return redirect(url_for('change_password'))

        # Retrieve the user's information from the database using the username
        conn = sqlite3.connect('mydb.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()

        # Check if the current password matches the stored password
        if user and user[2] == current_password:  # Assuming the password is stored in the third column
            # Update the password in the database
            c.execute('UPDATE users SET password = ? WHERE username = ?', (new_password, username))
            conn.commit()
            conn.close()
            flash('Password updated successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid current password!', 'error')
            conn.close()
            return redirect(url_for('change_password'))

    return render_template('change_password.html')

@app.route('/format_data', methods=['POST'])
def format_data():
    # Connect to the database
    conn = sqlite3.connect('mydb.db')
    cursor = conn.cursor()

    # Perform the data formatting operation
    try:
        cursor.execute('DELETE FROM detected_frames')
        conn.commit()
        flash('Data formatted successfully!', 'success')
    except Exception as e:
        conn.rollback()
        flash('Error formatting data: ' + str(e), 'error')
    finally:
        conn.close()

    # Redirect back to the participants page
    return redirect(url_for('participants'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port='5000')