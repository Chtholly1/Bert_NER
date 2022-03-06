#coding:utf-8
import re
import json
import copy
import sys

color_list = ['黑色', '黄色', '配色', '蓝灰色', '茶色', '银色', '极光银', '白色', '白', '颜色精致', '颜色', '灰白', '咖啡色', '金色', '青色', '双拼色', '绿色', '米色', '黑', '浅蓝色', '棕色', '浅蓝', '灰色', '粉', '玄武灰', '苍穹灰', '颜色比较少', '银白色', '玛瑙红', '红色', '蓝色', '灰', '粉红', '颜色比较多', '黑红色']
appearance_list = ['白色', '青色', '绿色', '橙色', '暗黑色', '暗红', '蓝白色', '红', '银色', '白', '戴茶色', '流光金', '玄武灰', '蓝色', '黑色', '黄色', '暗银色', '黑', '红黑', '粉色', '香槟金', '灰色', '红色', '天蓝色', '红黑色', '黄', '茶色', '紫红色', '米白色', '金黄色', '颜色', '墨灰色', '棕色', '深紫色', '星光粽', '紫色']
space_list = ['后排空间', '前排空间', '储物空间', '乘坐空间', '内部空间', '车内空间', '头部空间', '腿部空间']
chair_list = ['电动座椅']
car_type_miss = r'(车型)'
r_ct_m = re.compile(car_type_miss)

ori_color_label_list = ['appearance', 'interior']

map_labels = {'car_sries':'car_series', 'confort':'comfort'}

map_entities = {'座椅':'comfort', '行车记录仪':'config', '越野车':'car_type', '电动座椅':'comfort', '座椅加热':'comfort', '抬头显示':'control'}

#ori_data = sys.argv[1]
#with open(ori_data) as f:
for line in sys.stdin:
    info = eval(line.strip())
    text = info['text']
    label_dict = info['label']

    for key in map_labels:
        if key in label_dict:
            label_dict[map_labels[key]] = label_dict[key]

    for key in map_labels:
        if key in label_dict:
            label_dict.pop(key)

    new_label_dict = copy.deepcopy(label_dict)
    if 'color' not in new_label_dict:
        new_label_dict['color'] = {}
    it_ct = r_ct_m.finditer(text)
        
    #car_type
    it_ct_flag = False
    for it in it_ct:
        it_ct_flag = True
        entity, loc = it.group(), list(it.span())
        if 'car_type' not in new_label_dict:
            new_label_dict['car_type'] = {entity:[loc]}
        else:
            if entity not in new_label_dict['car_type']:
                new_label_dict['car_type'][entity] = [loc]
            else:
                if loc not in new_label_dict['car_type'][entity]:
                    new_label_dict['car_type'][entity].append(loc)

    for key, val in label_dict.items():
        #color
        if key in ori_color_label_list:
            for entity, loc_list in val.items():
                if entity in color_list or entity in appearance_list:
                    new_label_dict[key].pop(entity)
                    if entity in new_label_dict['color']:
                        new_label_dict['color'][entity].extend(loc_list)
                    else:
                        new_label_dict['color'][entity] = loc_list
        #space
        #if key == 'space':
        for entity, loc_list in val.items():
            if key == 'space' and entity.startswith('空间'):
                for loc in loc_list:
                    if text[loc[0]-2:loc[0]+2] in space_list and loc[1] -loc[0]<=5:
                        new_entity = text[loc[0]-2:loc[1]]
                        new_loc = [loc[0]-2, loc[1]]
                        new_label_dict['space'][entity].remove(loc)
                        if not new_label_dict['space'][entity]:
                            new_label_dict['space'].pop(entity)
                        if new_entity in new_label_dict['space']:
                            new_label_dict['space'][new_entity].append(new_loc)
                        else:
                            new_label_dict['space'][new_entity] = [new_loc]
            elif key == 'comfort' and entity.startswith('座椅'):
                for loc in loc_list:
                    if text[loc[0]-2:loc[0]+2] in chair_list and loc[1] -loc[0]<=5:
                        new_entity = text[loc[0]-2:loc[1]]
                        new_loc = [loc[0]-2, loc[1]]
                        new_label_dict['comfort'][entity].remove(loc)
                        if not new_label_dict['comfort'][entity]:
                            new_label_dict['comfort'].pop(entity)
                        if new_entity in new_label_dict['comfort']:
                            new_label_dict['comfort'][new_entity].append(new_loc)
                        else:
                            new_label_dict['comfort'][new_entity] = [new_loc]

        #           
        for entity, loc_list in val.items():
            if entity in map_entities and key != map_entities[entity]:
                new_label_dict[key].pop(entity)
                if map_entities[entity] not in new_label_dict:
                    new_label_dict[map_entities[entity]] = dict()
                if entity not in new_label_dict[map_entities[entity]]:
                    new_label_dict[map_entities[entity]][entity] = loc_list
                else:
                    new_label_dict[map_entities[entity]][entity].extend(loc_list)
    if not new_label_dict['color']:
        new_label_dict.pop('color')   
    info['label'] = new_label_dict
    info['text'] = text.split('\t')[0]
    #info['label'] = label_dict
    print(json.dumps(info, ensure_ascii=False))
