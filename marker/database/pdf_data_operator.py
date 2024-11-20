import mysql.connector
from marker.database.mysqldatabase import MySQLDatabase
from marker.config_read import Config


class PDFDataOperator:

    def __init__(self, config_file):
        self.db = MySQLDatabase(config_file)
        self.config = Config(config_file)

        if self.config.is_dev_mode():
            # 测试表名
            self.table_name_pri_pdf = 't_pdf_info_copy1'
            self.table_name_sub_md = 't_pdf_md_info_copy1'
            self.table_name_pri_docx = 't_word_info_copy1'
            self.table_name_pri_docx_handler = 't_word_handler_info_copy1'
        else:
            # 正式表名
            self.table_name_pri_pdf = 't_pdf_info'
            self.table_name_sub_md = 't_pdf_md_info'
            self.table_name_pri_docx = 't_word_info'
            self.table_name_pri_docx_handler = 't_word_handler_info'

    def query_need_ocr_v1(self, ocr_type, max_record):
        try:
            select_query = f"SELECT * FROM {self.table_name_pri_pdf} WHERE FINISH_OCR <> %s AND DELETE_FLAG = '0' AND ID NOT IN (SELECT b.PDF_INFO_ID FROM {self.table_name_sub_md} b WHERE b.OCR_TYPE = %s AND b.DELETE_FLAG = '0') ORDER BY CREATE_TIME"
            if max_record > 0:
                select_query = select_query + " LIMIT 0, %s"
            self.db.connect()
            if max_record > 0:
                results = self.db.query(select_query, (9, ocr_type, max_record))
            else:
                results = self.db.query(select_query, (9, ocr_type))
            return results
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def query_need_ocr_v2(self, max_record):
        try:
            select_query = f"SELECT * FROM {self.table_name_pri_pdf} WHERE FINISH_OCR = %s AND DELETE_FLAG = '0' ORDER BY OCR_PRIORITY, CREATE_TIME"
            if max_record > 0:
                select_query = select_query + " LIMIT 0, %s"
            self.db.connect()
            if max_record > 0:
                results = self.db.query(select_query, (0, max_record))
            else:
                results = self.db.query(select_query, (0,))
            return results
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def query_need_fix_v1(self, ocr_types, max_record):
        try:
            ocr_type_where = ", ".join([f"'{value}'" for value in ocr_types])
            select_query = f"SELECT a.* FROM {self.table_name_sub_md} a, {self.table_name_pri_pdf} b WHERE a.OCR_TYPE IN ({ocr_type_where}) AND a.DELETE_FLAG = '0' AND a.PDF_INFO_ID = b.ID AND b.FINISH_OCR <> %s AND b.DELETE_FLAG = '0' ORDER BY a.PDF_INFO_ID, a.CREATE_TIME"
            if max_record > 0:
                select_query = select_query + " LIMIT 0, %s"
            self.db.connect()
            if max_record > 0:
                results = self.db.query(select_query, (9, max_record))
            else:
                results = self.db.query(select_query, (9,))
            return results
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def query_need_fix_v2(self, max_record):
        try:
            select_query = f"SELECT a.* FROM {self.table_name_sub_md} a, {self.table_name_pri_pdf} b WHERE a.PDF_INFO_ID = b.ID AND a.DELETE_FLAG = '0' AND b.DELETE_FLAG = '0' AND b.FINISH_OCR <> %s AND a.OCR_FIX = 1 AND a.FINISH_FIX = 0 ORDER BY b.OCR_PRIORITY, a.CREATE_TIME"
            if max_record > 0:
                select_query = select_query + " LIMIT 0, %s"
            self.db.connect()
            if max_record > 0:
                results = self.db.query(select_query, (9, max_record))
            else:
                results = self.db.query(select_query, (9,))
            return results
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def query_sub_finish_ocr(self, ocr_type, max_record):
        try:
            select_query = f"SELECT * FROM {self.table_name_sub_md} WHERE OCR_TYPE = %s AND DELETE_FLAG = '0' ORDER BY CREATE_TIME"
            if max_record > 0:
                select_query = select_query + " LIMIT 0, %s"
            self.db.connect()
            if max_record > 0:
                results = self.db.query(select_query, (ocr_type, max_record))
            else:
                results = self.db.query(select_query, (ocr_type,))
            return results
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def query_sub_finish_ocr_all(self, max_record):
        try:
            select_query = f"SELECT * FROM {self.table_name_sub_md} WHERE DELETE_FLAG = '0' ORDER BY CREATE_TIME"
            if max_record > 0:
                select_query = select_query + " LIMIT 0, %s"
            self.db.connect()
            if max_record > 0:
                results = self.db.query(select_query, (max_record,))
            else:
                results = self.db.query(select_query)
            return results
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def query_sub_finish_fix(self, ocr_types, max_record):
        try:
            ocr_type_where = ", ".join([f"'{value}'" for value in ocr_types])
            select_query = f"SELECT * FROM {self.table_name_sub_md} WHERE OCR_TYPE IN ({ocr_type_where}) AND FINISH_FIX = 1 AND DELETE_FLAG = '0' ORDER BY CREATE_TIME"
            if max_record > 0:
                select_query = select_query + " LIMIT 0, %s"
            self.db.connect()
            if max_record > 0:
                results = self.db.query(select_query, (max_record,))
            else:
                results = self.db.query(select_query)
            return results
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def query_sub_all_record(self, max_record):
        try:
            select_query = f"SELECT * FROM {self.table_name_sub_md} WHERE DELETE_FLAG = '0' ORDER BY PDF_INFO_ID, CREATE_TIME"
            if max_record > 0:
                select_query = select_query + " LIMIT 0, %s"
            self.db.connect()
            if max_record > 0:
                results = self.db.query(select_query, (max_record,))
            else:
                results = self.db.query(select_query)
            return results
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def get_sub_record_number(self, record_id):
        try:
            select_query = f"SELECT COUNT(*) AS COUNT FROM {self.table_name_sub_md} WHERE PDF_INFO_ID = %s"
            self.db.connect()
            results = self.db.query(select_query, (record_id,))
            if len(results) > 0:
                return results[0]['COUNT']
            else:
                return 1
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def get_sub_finish_ocr_number(self, record_id, ocr_types):
        try:
            ocr_type_where = ", ".join([f"'{value}'" for value in ocr_types])
            select_query = f"SELECT COUNT(*) AS COUNT FROM {self.table_name_sub_md} WHERE PDF_INFO_ID = %s AND OCR_TYPE IN ({ocr_type_where}) AND DELETE_FLAG = '0'"
            self.db.connect()
            results = self.db.query(select_query, (record_id,))
            if len(results) > 0:
                return results[0]['COUNT']
            else:
                return 0
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def insert_sub_finish_ocr(self, record_id, sub_record_id, ocr_type, md_title, md_file_path, md_file_name):
        try:
            insert_query = f"INSERT INTO {self.table_name_sub_md}(ID, PDF_INFO_ID, OCR_TYPE, OCR_FIX, FINISH_FIX, MD_TITLE, OCR_DATE, MD_FILE_DIR, MD_FILE_NAME, DELETE_FLAG, CREATE_TIME, UPDATE_TIME) VALUES (%s, %s, %s, 0, 0, %s, NOW(), %s, %s, '0', NOW(), NOW())"
            self.db.connect()
            record_id = self.db.insert(insert_query,
                                       (sub_record_id, record_id, ocr_type, md_title, md_file_path, md_file_name))
            return record_id
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def update_pri_finish_orc(self, reocrd_id, finish_ocr):
        try:
            update_query = f"UPDATE {self.table_name_pri_pdf} SET FINISH_OCR = %s, OCR_DATE = NOW() WHERE ID = %s"
            self.db.connect()
            rows = self.db.update(update_query, (finish_ocr, reocrd_id))
            return rows
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def update_pri_finish_orc_start(self, reocrd_id):
        try:
            update_query = f"UPDATE {self.table_name_pri_pdf} SET FINISH_OCR = '1' WHERE ID = %s"
            self.db.connect()
            rows = self.db.update(update_query, (reocrd_id,))
            return rows
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def update_pri_finish_orc_end(self, reocrd_id):
        try:
            update_query = f"UPDATE {self.table_name_pri_pdf} SET FINISH_OCR = '9', OCR_DATE = NOW() WHERE ID = %s"
            self.db.connect()
            rows = self.db.update(update_query, (reocrd_id,))
            return rows
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def update_sub_finish_fix(self, reocrd_id, ocr_type, md_path, md_file):
        try:
            update_query = f"UPDATE {self.table_name_sub_md} SET FINISH_FIX = 1, UPDATE_TIME = NOW() WHERE ID = %s"
            self.db.connect()
            rows = self.db.update(update_query, (ocr_type, md_path, md_file, reocrd_id))
            return rows
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def query_all_valid_docx(self, max_record):
        try:
            select_query = f"SELECT * FROM {self.table_name_pri_docx} WHERE confirm = %s AND delete_flag = '0' ORDER BY creat_time"
            if max_record > 0:
                select_query = select_query + " LIMIT 0, %s"
            self.db.connect()
            if max_record > 0:
                results = self.db.query(select_query, (1, max_record))
            else:
                results = self.db.query(select_query, (1,))
            return results
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def query_need_docx(self, max_record):
        try:
            select_query = f"SELECT * FROM {self.table_name_pri_docx} WHERE confirm = %s AND delete_flag = '0' AND id not in (SELECT DISTINCT(word_info_id) FROM {self.table_name_pri_docx_handler} WHERE delete_flag = '0') ORDER BY creat_time"
            if max_record > 0:
                select_query = select_query + " LIMIT 0, %s"
            self.db.connect()
            if max_record > 0:
                results = self.db.query(select_query, (1, max_record))
            else:
                results = self.db.query(select_query, (1,))
            return results
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def query_pri_docx_kg(self, max_record):
        try:
            select_query = f"SELECT * FROM {self.table_name_pri_docx} WHERE finish_kg = %s AND delete_flag = '0' AND id in (SELECT DISTINCT(word_info_id) FROM {self.table_name_pri_docx_handler} WHERE delete_flag = '0') ORDER BY creat_time"
            if max_record > 0:
                select_query = select_query + " LIMIT 0, %s"
            self.db.connect()
            if max_record > 0:
                results = self.db.query(select_query, (0, max_record))
            else:
                results = self.db.query(select_query, (0,))
            return results
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def insert_pri_docx_handler(self, record_id):
        try:
            update_query = f"INSERT INTO {self.table_name_pri_docx_handler}(word_info_id, delete_flag, update_time) VALUES (%s, '0', NOW())"
            self.db.connect()
            recoid_id = self.db.insert(update_query, (record_id, ))
            return record_id
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def update_pri_fix_file(self, reocrd_id, json_file, json_file_name):
        try:
            update_query = f"UPDATE {self.table_name_pri_docx} SET json_file = %s, json_file_name=%s, update_time = NOW() WHERE id = %s"
            self.db.connect()
            rows = self.db.update(update_query, (json_file, json_file_name, reocrd_id))
            return rows
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()

    def update_pri_docx_kg_finish(self, reocrd_id):
        try:
            update_query = f"UPDATE {self.table_name_pri_docx} SET finish_kg = %s, finish_kg_date = NOW() WHERE id = %s"
            self.db.connect()
            rows = self.db.update(update_query, (1, reocrd_id))
            return rows
        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        finally:
            self.db.close()
