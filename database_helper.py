import pyodbc
from datetime import datetime

def get_db_connection():
    """Establishes a connection to the local SQL Server using Windows Authentication."""
    conn_str = (
        r'DRIVER={ODBC Driver 17 for SQL Server};'
        r'SERVER=localhost\SQLEXPRESS;'
        r'DATABASE=SmartStudentMonitor;'
        r'Trusted_Connection=yes;'
    )
    try:
        conn = pyodbc.connect(conn_str)
        return conn
    except Exception as e:
        print(f"❌ Database Connection Error: {e}")
        return None

def log_engagement(student_id, face_detected):
    """Inserts a live camera tracking frame record into the SQL database."""
    conn = get_db_connection()
    if conn is None:
        return

    cursor = conn.cursor()
    query = """
        INSERT INTO EngagementLogs (StudentID, Timestamp, FaceDetected)
        VALUES (?, ?, ?)
    """
    current_time = datetime.now()
    # Convert Boolean True/False to SQL BIT 1/0
    bit_value = 1 if face_detected else 0 
    
    try:
        cursor.execute(query, (student_id, current_time, bit_value))
        conn.commit()
        print(f"💾 Logged to SQL -> Student: {student_id} | Face Present: {face_detected} at {current_time.strftime('%H:%M:%S')}")
    except Exception as e:
        print(f"❌ Failed to insert log: {e}")
    finally:
        conn.close()