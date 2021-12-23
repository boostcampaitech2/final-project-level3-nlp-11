from typing import List
from google.cloud import bigquery
from google.oauth2 import service_account


class Logger:
    def __init__(
        self, table_id: str, credential_json_path: service_account.Credentials
    ):
        self.table_id = table_id
        self.credential_json_path = credential_json_path

    def insert_log(self, json_data: List[dict]):
        print("start insert log")
        credentials = service_account.Credentials.from_service_account_file(
            filename=self.credential_json_path
        )
        client = bigquery.Client(credentials=credentials)
        table = bigquery.table.Table(self.table_id)
        client.insert_rows_json(table, json_data)
        print("end insert log")

