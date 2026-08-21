import pymysql

connection = pymysql.connect(
    host="localhost",
    user="root",
    password="neha@2002",
    database="emp",
    port=3306
)

print("Connected to MySQL successfully!")

cursor = connection.cursor()

cursor.execute("""
    SELECT
        e.employee_code,
        CONCAT(e.first_name, ' ', e.last_name) AS employee_name,
        d.department_name,
        j.job_title,
        s.base_salary
    FROM employees e
    JOIN departments d
        ON e.department_id = d.department_id
    JOIN jobs j
        ON e.job_id = j.job_id
    JOIN salaries s
        ON e.emp_id = s.emp_id
    ORDER BY e.emp_id
""")

rows = cursor.fetchall()

for row in rows:
    print(row)

cursor.close()
connection.close()
