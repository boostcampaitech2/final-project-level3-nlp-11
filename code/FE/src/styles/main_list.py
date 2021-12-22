def main_list_style():
    return f"""
            <style media="screen and (min-width:100px) and (max-width:900px)">
                    #context_img {{width:100%}}
                    #context {{width:100%;padding-top:10px}}
                    #title_context {{display:None}}
            </style>
            <style media="screen and (min-width:900px) and (max-width:1400px)">
                #context_img {{width:40%;float:left; height:100%}}
                #context {{width:57%; float:right;height:100%}}
            </style>
            <style media="screen and (min-width:1400px) and (max-width:1800px)">
                #context_img {{width:30%;float:left; height:100%}}
                #context {{width:67%; float:right;height:100%}}
            </style>
            <style media="screen and (min-width:1800px)">
                #context_img {{width:20%;float:left; height:100%}}
                #context {{width:77%; float:right;height:100%}}
            </style>
    """
