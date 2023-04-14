import cv2
import numpy as np
import pyautogui


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cv_shape(img, clarity_level=1000):
    def is_gray_scale_image(img):
        """
        判断一张图片是否为灰度图
        :param image_path: 图片路径
        :return: True为灰度图，False为彩色图
        """
        if len(img.shape) < 3 or img.shape[2] == 1:
            return True
        else:
            return False
    if (clarity_level <= 0):
        print("erro")
    else:
        if(is_gray_scale_image(img)):
            # 获取原始图片尺寸
            height, width = img.shape
        else:
            height, width, _ = img.shape
        # 计算缩小比例
        if width > height:
            scale_percent = clarity_level / width
        else:
            scale_percent = clarity_level / height

        # 缩小图片
        new_width = int(width * scale_percent)
        new_height = int(height * scale_percent)
        new_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return new_img


def weighted_gray(image):
    """
    加权平均值法灰度化
    :param image: numpy ndarray，彩色图像
    :return: numpy ndarray，灰度图像
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    """
    b, g, r = cv2.split(image)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = gray.astype(np.float32)/255.0
    return gray


def get_chess_table():
    """获取棋盘区域并创建掩码"""
    # 获取屏幕分辨率
    screenWidth, screenHeight = pyautogui.size()

    # 截取整个屏幕
    # screenshot返回的不是一个数组
    screen_cur = np.array(pyautogui.screenshot())
    gray = weighted_gray(screen_cur)
    template = cv2.imread("chess_table.jpg")
    template = weighted_gray(template)
    h, w = template.shape
    res = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    """画出来"""
    img2 = gray.copy()
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img2, top_left, bottom_right, 255, 2)
    print("棋盘位置：",top_left,bottom_right)
    locate = (top_left,bottom_right)
    # 创建mast
    mask = np.zeros(screen_cur.shape[:2], np.uint8)
    mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255                     #（Y轴，X轴）

    masked_img = cv2.bitwise_and(gray, gray, mask=mask)#与操作
    # cv_show("screem", cv_shape(masked_img))
    return locate, masked_img



