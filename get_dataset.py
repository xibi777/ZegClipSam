source_dir = '/media/data/ziqin/data_fss/VOC2012/ImageSets/FewShotSegmentation/'
# source_dir = '/media/data/ziqin/data_fss/coco2014/ImageSets/FewShotSegmentation/'
# source_file:val_supp_split_0_shot_1_seed0.txt
target_dir = '/media/data/ziqin/data_fss/VOC2012/ImageSets/CrossFewShotSegmentation/'
# target_dir = '/media/data/ziqin/data_fss/coco2014/ImageSets/CrossFewShotSegmentation/'

shot = 5
seed = 4

save_path = target_dir + 'voc_supp_shot_' + str(shot) + '_seed' + str(seed) + '.txt'
ff=open(save_path, 'a+')

for i in range(4):
    source_path = source_dir + 'val_supp_split_' + str(i) + '_shot_' + str(shot) + '_seed' + str(seed) + '.txt'
    f = open(source_path, encoding='utf-8')
    while True:
        filename = f.readline()
        if filename:
            ff.write(filename)
        else:
            break
        # print(filename)
f.close()
ff.close()


# shot = 5
# seed = 4
# source_path_split_0 = '/media/data/ziqin/data_fss/coco2014/ImageSets/FewShotSegmentation/val_supp_split_0_shot_' + str(shot) + '_seed' + str(seed) + '.txt'
# source_path_split_1 = '/media/data/ziqin/data_fss/coco2014/ImageSets/FewShotSegmentation/val_supp_split_1_shot_' + str(shot) + '_seed' + str(seed) + '.txt'
# source_path_split_2 = '/media/data/ziqin/data_fss/coco2014/ImageSets/FewShotSegmentation/val_supp_split_2_shot_' + str(shot) + '_seed' + str(seed) + '.txt'
# source_path_split_3 = '/media/data/ziqin/data_fss/coco2014/ImageSets/FewShotSegmentation/val_supp_split_3_shot_' + str(shot) + '_seed' + str(seed) + '.txt'


# target_dir = '/media/data/ziqin/data_fss/coco2014/ImageSets/CrossFewShotSegmentation/'
# save_path = target_dir + 'coco_supp_shot_' + str(shot) + '_seed' + str(seed) + '.txt'
# ff=open(save_path, 'a+')

# f0 = open(source_path_split_0, encoding='utf-8')
# f1 = open(source_path_split_1, encoding='utf-8')
# f2 = open(source_path_split_2, encoding='utf-8')
# f3 = open(source_path_split_3, encoding='utf-8')

# # total_num = int(80*shot)
# # each_num_split = int(20*shot)

# for j in range(1):
#     while True:
#         for i in range(shot):
#             filename0 = f0.readline()
#             if filename0:
#                 ff.write(filename0)
#             else:
#                 break

#         for i in range(shot):
#             filename1 = f1.readline()
#             if filename1:
#                 ff.write(filename1)
#             else:
#                 break

#         for i in range(shot):
#             filename2 = f2.readline()
#             if filename2:
#                 ff.write(filename2)
#             else:
#                 break

#         for i in range(shot):
#             filename3 = f3.readline()
#             if filename3:
#                 ff.write(filename3)
#             else:
#                 break
    

# f0.close()
# f1.close()
# f2.close()
# f3.close()
# ff.close()