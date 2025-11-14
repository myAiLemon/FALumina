import os
from PIL import Image

def resize_images_to_min_side_1024(input_dir):
    """
    将指定文件夹下的图片放大，使最短边为1024像素，并保存到输出文件夹
    
    参数:
        input_dir: 输入图片文件夹的路径
    """
    # 检查输入文件夹是否存在
    if not os.path.isdir(input_dir):
        print(f"错误：文件夹 '{input_dir}' 不存在，请检查路径是否正确")
        return
    
    # 创建输出文件夹（在输入文件夹下创建"resized_1024"子文件夹）
    output_dir = os.path.join(input_dir, "resized_1024")
    os.makedirs(output_dir, exist_ok=True)
    print(f"处理后的图片将保存到：{output_dir}")
    
    # 支持的图片格式（不区分大小写）
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_dir):
        # 获取文件路径和扩展名
        file_path = os.path.join(input_dir, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        # 跳过文件夹和非图片文件
        if os.path.isdir(file_path) or file_ext not in image_extensions:
            continue
        
        try:
            # 打开图片
            with Image.open(file_path) as img:
                # 获取原图尺寸
                width, height = img.size
                print(f"处理图片：{filename}（原尺寸：{width}x{height}）")
                
                # 计算最短边和缩放比例
                min_side = min(width, height)
                scale = 1024 / min_side  # 缩放比例（>1表示放大）
                
                # 计算新尺寸（四舍五入取整数）
                new_width = round(width * scale)
                new_height = round(height * scale)
                
                # 使用高质量插值方法放大图片（LANCZOS适用于缩小和放大，效果较好）
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # 保存处理后的图片（保持原格式）
                output_path = os.path.join(output_dir, filename)
                resized_img.save(output_path)
                
                print(f"已保存：{filename}（新尺寸：{new_width}x{new_height}）")
        
        except Exception as e:
            print(f"处理 {filename} 时出错：{str(e)}，已跳过该文件")
    
    print("所有图片处理完成！")

if __name__ == "__main__":
    # 输入文件夹路径（可替换为实际路径，例如："C:/Users/YourName/Pictures"）
    input_folder = "/root/"
    resize_images_to_min_side_1024(input_folder)