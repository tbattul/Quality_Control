# test_mysql.py

# test_mysql.py
try:
    import mysql.connector
    print("mysql-connector-python is working!")
except ImportError:
    print("mysql-connector-python is NOT working!")

try:
    import mysql_connector_repackaged
    print("mysql_connector_repackaged is working!")
except ImportError:
    print("mysql_connector_repackaged is NOT working!")
