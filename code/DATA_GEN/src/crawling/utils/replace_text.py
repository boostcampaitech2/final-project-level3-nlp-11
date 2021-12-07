import re


class ReplaceText:
    def __init__(self) -> None:
        pass

    def get_convert_text(self, context):
        context = context.replace("\n", "")
        context = context.replace("&nbsp;", "")
        # 해쉬태그 and @태그 제거
        context = re.sub(r"""(#|@)[ㄱ-ㅎ|ㅏ-ㅣ|가-힣a-zA-Z]*""", "", context)
        # 공백 2개 한개로 변경
        context = re.sub(r""" +(?= )""", "", context)
        # URL
        context = re.sub(
            r"""https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)""",
            "",
            context,
        )
        # email
        context = re.sub(
            r"""([0-9a-zA-Z]([-_\.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_\.]?[0-9a-zA-Z])*\.[a-zA-Z]{2,3})""",
            "",
            context,
        )
        # 전화번호
        context = re.sub(r"""\d{3}-\d{3,4}-\d{4}""", "", context)
        # dict 형식 제거
        context = re.sub(r"""{[^}]*}""", " ", context)
        # 온점 후 띄어쓰기 강제
        # context = re.sub(r"""\.(?=[^ ])""", ". ", context)
        # 아래 나온 문자외 전부 제거
        context = re.sub(r"""[^0-9a-zA-Zㄱ-ㅎ|ㅏ-ㅣ|가-힣 ().,?!]""", "", context)
        # ㅋㅋ,ㅎㅎ,ㅜㅜ,ㅠㅠ,..,!!,??,~~ 2개로 제한
        context = re.sub(r"""(ㅋ[ㅋ]+)""", "ㅋㅋ", context)
        context = re.sub(r"""(ㅎ[ㅎ]+)""", "ㅎㅎ", context)
        context = re.sub(r"""(ㅜ[ㅜ]+)""", "ㅜㅜ", context)
        context = re.sub(r"""(ㅠ[ㅠ]+)""", "ㅠㅠ", context)
        context = re.sub(r"""(\.[.]+)""", ".. ", context)
        context = re.sub(r"""(![!]+)""", "! ", context)
        context = re.sub(r"""(\?[?]+)""", "? ", context)
        context = re.sub(r"""(~[~]+)""", "~~ ", context)
        return context
