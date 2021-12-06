import re
from bs4 import BeautifulSoup
from utils.replace_text import ReplaceText

class BlogParser:
    def __init__(self,response,url) -> None:
        self.response = response
        self.url = url
        
    def get_result(self):
        response = self.response
        url = self.url
        soup = BeautifulSoup(response.text,"lxml")
        text = None
        if soup.find("div", attrs={"class": "se-component se-text se-l-default"}):
            text = self.blog_type1(soup)
        elif soup.find("div", attrs={"class": "post_ct"}):
            text = self.blog_type2(soup)
        elif re.findall(
            r"""(<(p|h[0-7]) class="se_textarea".*<\/(p|h[0-7])>)""", response.text
        ):
            text = self.post_type1(response)
        elif re.findall(r"""http:\/\/cafe.naver.com.*""", url):
            return None, url
        else:
            print(f"\nError {url}")
            return None, url
        text = ReplaceText().get_convert_text(text)
        return text, url

    def blog_type1(self,soup):
        contexts = soup.find_all(
            "div", attrs={"class":"se-component se-text se-l-default"}
        )

        def map_fun(x):
            return x.get_text().replace("\n","").replace("\u200b","")

        contexts = list(map(map_fun,contexts))
        text = " ".join(contexts)
        return text
    
    def blog_type2(self,soup):
        text = soup.find("div", attrs={"class":"post_ct"}).get_text()
        return text

    def post_type1(self,response):
        contexts = re.findall(
            r"""(<(p|h[0-7]) class="se_textarea".*<\/(p|h[0-7])>)""", response.text
        )
        def map_fun(x):
            return re.sub(r"""<[^>]*>""","",x[0])
        contexts = list(map(map_fun,contexts))
        text = " ".join(contexts)
        return text