import re


class ReplaceText:
    def __init__(self) -> None:
        pass

    def get_convert_text(self, context):

        context_pattern = {
            r"""&nbsp;""": "",
            r"""\u200b""": "",
            r"""…""": "...",
            r"""\ufeff""": "",
            r""" +(?= )""": "",
            r"""\(Google 번역 제공\)""": "",
            r"""\(원문\).*""": "",
            r"""([0-9a-zA-Z]([-_\.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_\.]?[0-9a-zA-Z])*\.[a-zA-Z]{2,3})""": "",
            r"""https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)""": "",
            r"\<저작권자(\(c\)|ⓒ|©|\(Copyright\)|(\(c\))|(\(C\))).+?\>": "",
            r"저작권자\(c\)|ⓒ|©|(Copyright)|(\(c\))|(\(C\))": "",
            r"\(출처 ?= ?.+\) |\(사진 ?= ?.+\) |\(자료 ?= ?.+\)| \(자료사진\) |사진=.+기자 ": "",
            r"""\d{3}-\d{3,4}-\d{4}""": "",
            r"""{[^}]*}""": "",
            r"""(#|@)[ㄱ-ㅎ|ㅏ-ㅣ|가-힣a-zA-Z]*""": "",
            r"""(ㅋ[ ㅋ]+)""": "ㅋㅋ",
            r"""(ㅎ[ ㅎ]+)""": "ㅎㅎ",
            r"""(ㅜ[ ㅜ]+)""": "ㅜㅜ",
            r"""(ㅠ[ ㅠ]+)""": "ㅠㅠ",
            r"""(\.[ .]+)""": "..",
            r"""(![ !]+)""": "!!",
            r"""(\?[ ?]+)""": "??",
            r"""(~[ ~]+)""": "~~",
            r"""[^0-9a-zA-Zㄱ-ㅎ|ㅏ-ㅣ|가-힣 ().,?!]""": "",
            r"""\n""": "",
        }

        for i in context_pattern.keys():
            try:
                context = re.sub(i, context_pattern[i], context)
            except:
                context
        if type(context) == type(0.0):
            return None

        tmp = re.findall(r"""[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]""", context)
        if not tmp:
            return None

        return context
