import face_recognition
from PIL import Image

# 路径设置
image_path = "person.jpg"         # 原始图片
emoji_path = "emoji.jpg"             # 用于替换人脸的 Emoji 图像
output_path = "emoji_faces.jpg"      # 输出路径

# 加载原始图像并检测人脸
image = face_recognition.load_image_file(image_path)
face_locations = face_recognition.face_locations(image)

# 转为 PIL 图像用于绘制
pil_image = Image.fromarray(image)
emoji_image = Image.open(emoji_path).convert("RGBA")  # 读取 emoji 图片（建议使用 PNG，带透明背景）

# 替换每一张人脸
for face_location in face_locations:
    top, right, bottom, left = face_location

    face_width = right - left
    face_height = bottom - top

    # 缩放 emoji 图片以适应人脸大小
    resized_emoji = emoji_image.resize((face_width, face_height))

    # 粘贴（带 alpha 通道即可自动保留透明背景）
    pil_image.paste(resized_emoji, (left, top), resized_emoji)

# 保存输出图像
pil_image.save(output_path)
print(f"已保存图片：{output_path}")
