import json


def parseFile(file_Path):  # filePath=Processed_Contracts
    with open(file_Path, 'rb') as dict1:
        all_dict=json.load(dict1)
    root_list=all_dict['annotations']
    print(1)

    cset=set()
    for dicti in root_list:
        classid=dicti["category_id"]
        cset.add(classid)

    dic={}
    for i in cset:
        dic[i]=0

    for k in root_list:
        classid=k["category_id"]
        dic[classid]+=1
    #print(dic)

    end_dict=sorted(dic.items(),key=lambda x:x[1],reverse=False)  # 按字典集合中，每一个元组的第二个元素排列。
    #print(end_dict) #tuple

    class_id=[]  #x
    class_num=[] #y
    for tup in end_dict:
        class_id.append(tup[0])
        class_num.append(tup[1])





if __name__ == '__main__':
    file_Path = '/dataset/cv/cls/herbarium-2022-fgvc9/train_metadata.json'
    parseFile(file_Path)
