from PIL import Image
import os

# Đường dẫn thư mục chứa ảnh
folder_path = './attention_maps/'

# Danh sách để lưu các ảnh
images = []

# Lấy tất cả file ảnh có pattern bắt đầu với 'attn_epoch_10'
for filename in sorted(os.listdir(folder_path)):  # Sắp xếp để có thứ tự nhất quán
    if filename.startswith('attn_epoch_10_batch_32_head_3') :
        file_path = os.path.join(folder_path, filename)
        
        # Mở ảnh và thêm vào danh sách
        img = Image.open(file_path)
        images.append(img)

# Tạo ảnh lớn
if images:
    # Tính số hàng và cột (có thể điều chỉnh số cột theo ý muốn)
    num_images = len(images)
    cols = 4  # Số ảnh mỗi hàng
    rows = (num_images + cols - 1) // cols  # Làm tròn lên để đủ chỗ

    # Lấy kích thước của ảnh đầu tiên (giả sử tất cả ảnh có cùng kích thước)
    img_width = images[0].width
    img_height = images[0].height

    # Tính kích thước tổng
    total_width = img_width * cols
    total_height = img_height * rows

    # Tạo ảnh lớn với nền trắng
    combined_img = Image.new('RGB', (total_width, total_height), 'white')

    # Đặt các ảnh nhỏ vào ảnh lớn
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * img_width
        y = row * img_height
        combined_img.paste(img, (x, y))

    # Lưu ảnh lớn
    output_path = os.path.join(folder_path, 'combined_attention_maps_32_3.png')
    combined_img.save(output_path)
    print(f"Ảnh đã được gộp và lưu tại: {output_path}")
    print(f"Kích thước ảnh cuối: {total_width}x{total_height}")
    print(f"Số ảnh đã gộp: {num_images}")
else:
    print("Không tìm thấy ảnh nào để gộp!")
