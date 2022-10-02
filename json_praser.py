import json


def split_data_8_2(file_Path):  # filePath=Processed_Contracts
    with open(file_Path, 'rb') as dict1:
        all_dict=json.load(dict1)

    root_list=all_dict['annotations']
    file_list=all_dict['images']
    root_list.append(root_list[0])
    print(len(root_list))
    train_list_8=[]
    test_list_2=[]

    print(root_list[1])

    preid=0
    sum=0

    pre=0
    end=0

    # t=0
    # root_list
    total=0
    for i in range(839773):
        class_id=root_list[i]["category_id"]
        if(class_id!=preid or i==839772):
            # t+=1
            preid=class_id
            train_num=int(sum*0.8)

            mid=pre+train_num
            k=pre
            total+=sum

            #exit()
            while(k<i):
                if(k<mid):
                    root_list[k]["file_name"]=file_list[k]['file_name']
                    train_list_8.append(root_list[k])
                else:
                    root_list[k]["file_name"] = file_list[k]['file_name']
                    test_list_2.append(root_list[k])
                k+=1

            sum=0
            pre=i
            end=i

        else:
            sum+=1
            end+=1
        if (i == 839772):
            break

    return train_list_8,test_list_2

def split_mid_8(train_list):
    train_list.append(train_list[0])
    print(len)
    mid_list_1 = []
    mid_list_2 = []

    preid = 0
    sum = 0

    pre = 0
    end = 0

    total = 0
    for i in range(653139):
        class_id = train_list[i]["category_id"]
        if (class_id != preid or i == 653138):
            preid = class_id
            train_num = int(sum * 0.5)

            mid = pre + train_num
            k = pre
            total += sum

            while (k < i):
                if (k < mid):
                    mid_list_1.append(train_list[k])
                else:
                    mid_list_2.append(train_list[k])
                k += 1

            sum = 0
            pre = i
            end = i

        else:
            sum += 1
            end += 1
        if (i == 653138):
            break

    return mid_list_1, mid_list_2



if __name__ == '__main__':
    file_Path = '/dataset/cv/cls/herbarium-2022-fgvc9/train_metadata.json'
    split_data_8_2(file_Path)
