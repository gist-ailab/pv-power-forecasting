from notion.client import NotionClient
from notion.block import PageBlock, TextBlock, TodoBlock, CollectionViewBlock, ToggleBlock
from notion.block.collection import CollectionBlock

from datetime import datetime

token_v2 = 'v02%3Auser_token_or_cookies%3AaTxLfL8xY-UDZ5p2aAtxTbed5W-eemCIuWWFa3AVjCxLQgfgTCS5yMYnVajuXFmHjmztFI9m3uqFSELSHdAinhn73y1VrG0QztVbnHUEtsI0nwt9gd87Z-LrGGsev4uMUjtL'
client = NotionClient(token_v2=token_v2)

url = 'https://www.notion.so/gistailab/7047b728926d4603b29fd07b652e00f7?pvs=4'
page = client.get_block(url)
print("The title is:", page.title)

model_id_name='240103_1BtoG_01'

print()

    # def get_schema_todo():
    # return {
    #     # title 항상 존재 해야한다
    #     "title": {"name": "내용", "type": "title"},
    #     "complete": {"name": "체크박스", "type": "checkbox"},
    #     "priority": {"name": "셀릭트박스", "type": "select",
    #         "options": [
    #             {
    #                 "color": "red",
    #                 "id": "502c7016-ac57-413a-90a6-64afadfb0c44",
    #                 "value": "사과",
    #             },
    #             {
    #                 "color": "yellow",
    #                 "id": "59560dab-c776-43d1-9420-27f4011fcaec",
    #                 "value": "오렌지",
    #             },
    #             {
    #                 "color": "green",
    #                 "id": "57f431ab-aeb2-48c2-9e40-3a630fb86a5b",
    #                 "value": "수박",
    #             }
    #         ]
    #     }
    # }

def table_schema():
    return {
        'title': 'result',
    }
    
toggle = page.children.add_new(ToggleBlock, title=model_id_name)
table = toggle.children.add_new(CollectionViewBlock)

table.collection = client.get_collection(
    client.create_record(
        "collection", parent=table, schema=table_schema()
    )
)
table.title = model_id_name
table.views.add_new(view_type="table")
row = table.collection.add_row()


# ####

child = page.children[0]
row = child.collection.add_row(title='test')
row.children.add_new(ToggleBlock, title='hi')


table = row.children.add_new(CollectionViewBlock)
table.collection = client.get_collection(
    client.create_record(
        "collection", parent=table, schema=table_schema()
    )
)
table.title = model_id_name
table.views.add_new(view_type="table")


row = table.collection.add_row()
row.


result = row.children.add_new(CollectionViewBlock, title='he')
result


    # child = page.children.add_new(CollectionViewBlock)
    # child.collection = client.get_collection(
    #     client.create_record(
    #         "collection", parent=child, schema=get_schema_todo())
    #     )
    # child.title = '생성한 테이블'
    # child.views.add_new(view_type="table")

    # row = child.collection.add_row()
    # row.set_property('title', '추가된 아이템')
    # row.set_property('체크박스', True)
    # row.set_property('셀릭트박스', '사과')
    
    
