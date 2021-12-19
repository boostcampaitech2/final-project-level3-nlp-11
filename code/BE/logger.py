from typing import List
from google.cloud import bigquery
from google.oauth2 import service_account


log_format = ", ".join(
    [
        f"%({key})s"
        for key in sorted(["filename", "levelname", "name", "message", "created"])
    ]
)


class Logger:
    # TODO 검색 시점과 사용자가 평가하는 시점이 다름.
    # query에 대한 검색 결과를 우선 로깅하고, 다른 테이블에 사용자 평가 포함 테이블로 로깅하는 방법 고민 중.
    def __init__(
        self, table_id: str, credential_json_path: service_account.Credentials
    ):
        self.table_id = table_id
        self.credential_json_path = credential_json_path

    def insert_log(self, json_data: List[dict]):
        credentials = service_account.Credentials.from_service_account_file(
            filename=self.credential_json_path
        )
        client = bigquery.Client(credentials=credentials)
        table = bigquery.table.Table(self.table_id)
        client.insert_rows_json(table, json_data)

