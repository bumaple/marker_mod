import mysql.connector

from marker.config_read import Config

class MySQLDatabase:
    def __init__(self, config_file):
        self.config = Config(config_file)
        self.connection = None
        self.connect()

    def connect(self):
        if not self.connection or not self.connection.is_connected():
            self.connection = mysql.connector.connect(
                host=self.config.get_mysql_param('host'),
                port=self.config.get_mysql_param('port'),
                user=self.config.get_mysql_param('user'),
                password=self.config.get_mysql_param('password'),
                database=self.config.get_mysql_param('database')
            )
        return self.connection

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def query(self, query, params=None):
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            field_names = [i[0] for i in cursor.description]  # 获取字段名
            # 将字段名和值组合在一起，返回字典列表
            combined_results = [dict(zip(field_names, row)) for row in results]
            return combined_results
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            cursor.close()

    def insert(self, insert_query, data):
        try:
            cursor = self.connection.cursor()
            cursor.execute(insert_query, data)
            self.connection.commit()
            return cursor.lastrowid  # 返回最后插入行的ID
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
            self.connection.rollback()
        finally:
            cursor.close()

    def update(self, update_query, data):
        try:
            cursor = self.connection.cursor()
            cursor.execute(update_query, data)
            self.connection.commit()
            return cursor.rowcount  # 返回受影响的行数
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
            self.connection.rollback()
        finally:
            cursor.close()

if __name__ == "__main__":
    # 从配置文件中读取数据库配置
    db = MySQLDatabase("config.ini")

    # 查询数据示例
    select_query = "SELECT * FROM your_table WHERE id = %s"
    result = db.query(select_query, (1,))
    print("查询结果:", result)

    # 插入数据示例
    insert_query = "INSERT INTO your_table (name, age) VALUES (%s, %s)"
    new_id = db.insert(insert_query, ("Alice", 30))
    print("插入的新记录ID:", new_id)

    # 修改数据示例
    update_query = "UPDATE your_table SET age = %s WHERE name = %s"
    affected_rows = db.update(update_query, (31, "Alice"))
    print("受影响的行数:", affected_rows)

    # 关闭数据库连接
    db.close()
