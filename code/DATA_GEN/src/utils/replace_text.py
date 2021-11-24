import re


class ReplaceText:
    def __init__(self) -> None:
        pass

    def get_convert_text(self, context):
        context = context.replace("\n", "")
        context = context.replace("&nbsp;", "")
        # 해쉬태그 제거(한글만)
        context = re.sub(r"""#[ㄱ-ㅎ|ㅏ-ㅣ|가-힣 ]*""", "", context)
        # 공백 2개 한개로 변경
        context = re.sub(r""" +(?= )""", "", context)
        # ULR
        context = re.sub(r"""https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)""", "", context)
        # dict 형식 제거
        context = re.sub(r"""{[^}]*}""", " ", context)
        # 온점 후 띄어쓰기 강제
        # context = re.sub(r"""\.(?=[^ ])""", ". ", context)
        # 아래 나온 문자외 전부 제거
        context = re.sub(r"""[^0-9a-zA-Zㄱ-ㅎ|ㅏ-ㅣ|가-힣 ().,?!]""", "", context)
        return context