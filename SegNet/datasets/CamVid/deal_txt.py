#!/usr/bin/python


import cv2


def deal_txt():
    txts = ['./train_org.txt', './test_org.txt', './val_org.txt']
    for fn in txts:
        with open(fn, 'r') as fr:
            all_lines_before = fr.read().splitlines()
            all_lines_after = [el.split(' ')[0].split('/')[-1] for el in all_lines_before]
            new_fn = './' + fn.split('.')[-2].split('_')[0] + '.txt'
            print('convertint {} to {}'.format(fn, new_fn))

            with open(new_fn, 'w') as fw:
                [fw.write(el + '\n') for el in all_lines_after]


def show_pic():
    with open('./train.txt', 'r') as fr:
        fn = fr.read().splitlines()[0]
    img = cv2.imread('./train/' + fn)
    label = cv2.imread('./trainannot/' + fn)
    label = label * 50
    cv2.imshow('img', img)
    cv2.imshow('label', label)
    cv2.waitKey()


if __name__ == '__main__':
    show_pic()
