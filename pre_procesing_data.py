
from tokenization import FullTokenizer

tokenizer=FullTokenizer("./bert_model/chinese_L-12_H-768_A-12/vocab.txt")


class DataProcess:
    @staticmethod
    def process_from_p_2_train(filepath_r,filepath_w):
        fw=open(filepath_w,mode="w",encoding="utf-8")
        with open(filepath_r,mode="r",encoding="utf-8") as fr:
            c=0
            for line in fr:
                pre_all,p=line.strip().split("\t")
                text_dict=eval(pre_all)
                subject=""
                object=""
                for spo in text_dict["spo_list"]:
                    if spo["predicate"]==p:
                        subject=spo["subject"]
                        object=spo["object"]
                # print(subject,object)
                # 提取出这句话
                text_list=[]
                text=""
                index=0
                for word_pos in text_dict["postag"]:
                    text_list.append([word_pos["word"],word_pos["pos"],index,index+len(word_pos["word"])])
                    index+=len(word_pos["word"])
                    text+=word_pos["word"]

                # 找到主客体的位置
                sub_start=text.find(subject)
                sub_end=sub_start+len(subject)
                obj_start=text.find(object)
                obj_end = obj_start + len(object)
                ## 加入标签
                for token in text_list:
                    L = token[2]
                    R = token[3]
                    if (L>=sub_start and L<sub_end) or (sub_start>=L and sub_start<R):
                        token.append("I-SUB")
                    elif (L >= obj_start and L < obj_end) or (obj_start >= L and obj_start < R):
                            token.append("I-OBJ")
                    else:
                        token.append("o")

                # 修改第一个I-SUB
                for token in text_list:
                    if token[-1]=="I-SUB":
                        token[-1]="B-SUB"
                        break
                # 修好最后一个I-SUB
                for i in range(len(text_list)-1,-1,-1):
                    if text_list[i][-1]=="I-SUB":
                        text_list[i][-1]="E-SUB"
                        break

                # 修改第一个i-OBJ
                for token in text_list:
                    if token[-1]=="I-OBJ":
                        token[-1]="B-OBJ"
                        break
                # 修好最后一个I-OBJ
                for i in range(len(text_list)-1,-1,-1):
                    if text_list[i][-1]=="I-OBJ":
                        text_list[i][-1]="E-OBJ"
                        break

                for token in text_list:
                    word = "".join(tokenizer.tokenize(token[0]))
                    if word=="":
                        continue
                    tmp=[word,token[1],token[-1]]
                    fw.write("\t".join(tmp)+"\n")
                fw.write(p+"\n")
                fw.write("\n")

                if c%1000==0:
                    print("语料处理中，，，，",c)
                c+=1




if __name__=="__main__":
    DataProcess.process_from_p_2_train("./data/dev_data.p","./data/dev.train")
    print("dev,数据处理完毕")
    DataProcess.process_from_p_2_train("./data/train_data.p","./data/train.train")
    print("train","数据处理完毕")