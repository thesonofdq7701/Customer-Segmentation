import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from scipy.stats import boxcox

# Using menu
st.title("DATA SCIENCE PROJECT")
menu = ["Introduction", "My Project", "Predict New Data"]
choice = st.sidebar.selectbox('Danh mục', menu)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data = pd.read_csv('D:\CourseDataScience\LDS0\project\project_3\model_project_3\OnlineRetail.csv', encoding='latin1')
data_positive = pd.read_csv('D:\CourseDataScience\LDS0\project\project_3\model_project_3\data_positive.csv', encoding='latin1')
data_RFM_raw = pd.read_csv('D:\CourseDataScience\LDS0\project\project_3\model_project_3\data_RFM_raw.csv', encoding='latin1')
data_RFM_mean = pd.read_csv('D:\CourseDataScience\LDS0\project\project_3\model_project_3\data_RFM_mean.csv', encoding='latin1')
data_RFM_with_id = pd.read_csv('D:\CourseDataScience\LDS0\project\project_3\model_project_3\data_RFM_with_id.csv', encoding='latin1')
data_RFM_train = pd.read_csv('D:\CourseDataScience\LDS0\project\project_3\model_project_3\data_RFM_train.csv', encoding='latin1')

# Model 
file_name_1 = 'D:/CourseDataScience/LDS0/project/project_3/model_project_3/rfm_model.sav'
loaded_model_1= pickle.load(open(file_name_1, 'rb' ))

filename_2 = 'D:/CourseDataScience/LDS0/project/project_3/model_project_3/rfm_hierarchical_model.sav'
loaded_model_2 = pickle.load(open(filename_2, 'rb' ))

# Box-Cox transform function
def boxcox_transform(data, feature_name):
    transformed_data, lambda_ = boxcox(data[feature_name])
    data[feature_name] = transformed_data
    return data,lambda_

# Apply Box-Cox transform to each feature


# Khởi tạo Quantile Transformer
def quantile_transform(data):
    quantile_transformer = QuantileTransformer(output_distribution='uniform')
    return quantile_transformer.fit_transform(data[['Recency', 'Frequency', 'Monetary']])

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if choice == 'Introduction':    
    st.subheader("GIỚI THIỆU")  
    st.subheader("Customer Segmentation") 
    # Hiển thị hình ảnh từ file
    image = open("D:\CourseDataScience\LDS0\project\project_3\GUI_Project_3\Customer_Segmentation.png", "rb").read()
    st.image(image, use_column_width=True)
    
    st.write("## I. Mục tiêu")  
    st.write("""### **Xây dựng một hệ thống phân nhóm khách hàng dựa trên thông tin do công ty cung cấp nhằm xác định các nhóm khách hàng tiềm năng để có thể xây dựng chiến lược kinh doanh một cách hợp lí.**""")
    st.write("## II. Mô hình RFM trong Customer Segnmentation")
    st.write("""### **Mô hình RFM là một phương pháp phổ biến được sử dụng trong phân đoạn khách hàng (customer segmentation), một phần của chiến lược quản lý mối quan hệ khách hàng (Customer Relationship Management - CRM). RFM là viết tắt của Recency, Frequency, và Monetary:**

### -   **Recency (R):** Đo lường thời gian kể từ lần giao dịch cuối cùng của khách hàng. Mức độ recency cao hơn thường cho thấy một khách hàng gần đây đã tương tác với doanh nghiệp hơn, điều này có thể biểu thị sự tương tác tích cực hoặc cần phải quan tâm hơn đến việc giữ chân khách hàng.

### -   **Frequency (F):** Đo lường số lượng các giao dịch mà một khách hàng đã thực hiện trong một khoảng thời gian nhất định. Mức độ frequency cao hơn thường cho thấy một khách hàng có độ trung thành cao với doanh nghiệp và có thể tạo ra doanh thu ổn định.

### -   **Monetary (M):** Đo lường giá trị tổng của các giao dịch mà một khách hàng đã thực hiện trong một khoảng thời gian nhất định. Mức độ monetary cao hơn thường cho thấy một khách hàng là một nguồn thu nhập quan trọng cho doanh nghiệp.

### **Bằng cách kết hợp các chỉ số này, mô hình RFM phân loại khách hàng thành các nhóm dựa trên hành vi mua hàng của họ, giúp doanh nghiệp hiểu rõ hơn về đặc điểm của từng nhóm và tạo ra các chiến lược tiếp thị và chăm sóc khách hàng hiệu quả.**
""")
    st.write("## III. Dữ liệu")
    # Hiển thị 10 dòng đầu của dữ liệu
    st.write("### **Đây là một phần của dữ liệu:**")
    st.dataframe(data.head(10))

    st.write("## IV. Nhóm")
    st.write("### - Nguyễn Thế Sơn")
    st.write("### - Nguyễn Trung Sơn")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
elif choice == 'My Project':    

    st.subheader("I.Giới thiệu về dữ liệu")
    st.write("### **Dữ liệu gốc:**")
    st.dataframe(data.sample(5))
    st.dataframe(data.describe())
    st.write("### **Dữ liệu RFM:**")
    st.dataframe(data_RFM_raw.sample(5))

    st.subheader("II.Trực quan hóa dữ liệu")
    st.write("### **1.Doanh thu theo quốc gia**")
    image = open("D:/CourseDataScience/LDS0/project/project_3/model_project_3/Truc_quan_hoa/revenue_by_country.png", "rb").read()
    st.image(image, use_column_width=True)
    st.write("""### **Nhận xét:**
#### - **Các khách hàng lớn thường tập trung ở UK, đây là một thị trường lớn cần tập trung vào quốc gia này.**""")
    st.write("### **2.Doanh thu theo tháng**")
    image = open("D:/CourseDataScience/LDS0/project/project_3/model_project_3/Truc_quan_hoa/revenue_by_month.png", "rb").read()
    st.image(image, use_column_width=True)
    st.write("""### **Nhận xét:**
#### - **Nếu chỉ xét ở năm 2011, doanh thu 8 tháng đầu năm khá ổn định ở mức 600.000\$ đến 800.000\$.**
#### - **Trong khi đó, từ tháng 9 đến tháng 11 doanh thu tăng vọt lên mức 1.000.000$. Có lẽ, cuối năm có nhiều các ngày lễ, chương trình khuyễn mãi khiến doanh thu tăng vọt**
#### - **Tháng 12 cuối năm doanh thu thấp nhất trong cả năm.**
#### - **Từ đó, ta có thể thấy được các tháng cuối năm sẽ là các tháng cực kì hot, cần tập trung vào những tháng cuối năm để có thể tăng thêm doanh số.**""")
    
    st.subheader("III.Customer Segmentation")
    st.write("### **1.Kết quả**")
    st.dataframe(data_RFM_mean.head(10))
    st.write("### **2.Trực quan hóa**")
    image = open("D:\\CourseDataScience\\LDS0\\project\\project_3\\model_project_3\\Unsupervised_Segments.png", "rb").read()
    st.image(image, use_column_width=True)
    image = open("D:/CourseDataScience/LDS0/project/project_3/model_project_3/rfm_scatter_unsupervised.png", "rb").read()
    st.image(image, use_column_width=True)
    st.write("""### **Nhận xét:**
#### - **Kết quả của phương pháp phân loại sử dụng tập luật dựa trên dữ liệu RFM tương đối tốt. Số lượng khách hàng ở mỗi nhóm là khá đồng đều.**
""")
    st.write("""### **Kết luận:**
### **1.Regular Customers**
#### - **Nhóm này có RecencyMean thấp, FrequencyMean và MonetaryMean trung bình.**
#### - **Thường xuyên tương tác với nhóm khách hàng này để duy trì nguồn khách hàng này.**
#### - **Cung cấp ưu đãi đặc biệt hoặc chương trình khuyễn mãi để khuyến khích họ mua sắm thường xuyên hơn.**
### **2.Loyal Customers:**
#### - **Nhóm này có RecencyMean cao, FrequencyMean và MonetaryMean trung bình.**
#### - **Ta nên tri ân khách hàng bằng các chương trình khách hàng trung thành và ưu đãi đặc biệt.**
#### - **Cung cấp ưu đãi đặc biệt hoặc chương trình khuyễn mãi để khuyến khích họ mua sắm thường xuyên hơn.**
### **3.Low-Value Customers:**
#### - **Nhóm này có RecencyMean và MonetaryMean thấp, FrequencyMean cũng thấp.**
#### - **Ta nên Nắm bắt cơ hội để tăng giá trị từ nhóm này bằng cách tăng cường tiếp thị và quảng cáo.**
#### - **Cung cấp các gói ưu đãi hoặc sản phẩm giá trị thấp để thu hút họ mua sắm hơn.**   
### **4.Big Spenders:**
#### - **Nhóm này có RecencyMean và FrequencyMean thấp, nhưng MonetaryMean cao.**
#### - **Tạo ra các gói sản phẩm hoặc dịch vụ cao cấp hơn để thu hút sự quan tâm của họ.**
#### - **Cung cấp dịch vụ hỗ trợ cao cấp và tùy chỉnh để tăng cường trải nghiệm mua sắm của họ.**    
### **2.Inactive Customers:**
#### - **Nhóm này có RecencyMean cao, FrequencyMean và MonetaryMean thấp.**
#### - **Tìm hiểu lý do tại sao họ không hoạt động và thử nghiệm các chiến lược tái kích hoạt.**
#### - **Gửi các thông điệp và ưu đãi đặc biệt để khuyến khích họ quay lại.**           
""")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

elif choice == 'a':
    # Sử dụng các điều khiển nhập
    # 1. Text
    st.subheader("1. Text")
    name = st.text_input("Enter your name")
    st.write("Your name is", name)
    # 2. Slider
    st.subheader("2. Slider")
    age = st.slider("How old are you?", 1, 100, 20)
    st.write("I'm", age, "years old.")
    # 3. Checkbox
    st.subheader("3. Checkbox")
    # agree = st.checkbox("I agree")
    if st.checkbox("I agree"):
        st.write("Great!")
    # 4. Radio
    st.subheader("4. Radio")
    status = st.radio("What is your status?", ("Active", "Inactive"))
    st.write("You are", status)
    # 5. Selectbox
    st.subheader("5. Selectbox")
    occupation = st.selectbox("What is your occupation?", ["Student", "Teacher", "Others"])
    st.write("You are a", occupation)
    # 6. Multiselect
    st.subheader("6. Multiselect")
    location = st.multiselect("Where do you live?", ("Hanoi", "HCM", "Danang", "Hue"))
    st.write("You live in", location)
    # 7. File Uploader
    st.subheader("7. File Uploader")
    file = st.file_uploader("Upload your file", type=["csv", "txt"])
    if file is not None:
        st.write(file)    
    # 9. Date Input
    st.subheader("9. Date Input")
    date = st.date_input("Pick a date")
    st.write("You picked", date)
    # 10. Time Input
    st.subheader("10. Time Input")
    time = st.time_input("Pick a time")
    st.write("You picked", time)
    # 11. Display JSON
    st.subheader("11. Display JSON")
    json = st.text_input("Enter JSON", '{"name": "Alice", "age": 25}')
    st.write("You entered", json)
    # 12. Display Raw Code
    st.subheader("12. Display Raw Code")
    code = st.text_area("Enter code", "print('Hello, world!')")
    st.write("You entered", code)
    # Sử dụng điều khiển submit
    st.subheader("Submit")
    submitted = st.button("Submit")
    if submitted:
        st.write("You submitted the form.")
        # In các thông tin phía trên khi người dùng nhấn nút Submit
        st.write("Your name is", name)
        st.write("I'm", age, "years old.")
        st.write("You are", status)
        st.write("You are a", occupation)
        st.write("You live in", location)
        st.write("You picked", date)
        st.write("You picked", time)
        st.write("You entered", json)
        st.write("You entered", code)


elif choice=='Predict New Data':
    st.write("##### 1. Some data")
    # Chọn nhập mã khách hàng hoặc nhập thông tin khách hàng vào dataframe
    st.write("##### 1. Chọn cách nhập thông tin khách hàng")
    type = st.radio("Chọn cách nhập thông tin khách hàng", options=["Nhập mã khách hàng", 
                                                                    "Nhập thông tin khách hàng vào dataframe"])
    if type == "Nhập mã khách hàng":
        # Nếu người dùng chọn nhập mã khách hàng
        st.subheader("Nhập mã khách hàng")
        # Tạo điều khiển để người dùng nhập mã khách hàng
        customer_id = st.text_input("Nhập mã khách hàng")
        if customer_id.isdigit() == True:
            if not data_positive[data_positive['CustomerID'] == int(customer_id)].empty:
                data_predict = data_positive[data_positive['CustomerID'] == int(customer_id)]
                data_RFM_predict = data_RFM_with_id[data_RFM_with_id['CustomerID'] == int(customer_id)].drop(columns=['CustomerID'])
                st.write("#### **Thông tin khách hàng:**")
                st.dataframe(data_predict)
                result = loaded_model_1.predict(data_RFM_predict)
                if result == 0:
                    st.write("#### **Khách hàng có id**", customer_id, "**được phân vào nhóm Regular Customers.**")
                elif result == 1:
                    st.write("#### **Khách hàng có id**", customer_id, "**được phân vào nhóm Loyal Customers.**")
                elif result == 2:
                    st.write("#### **Khách hàng có id**", customer_id, "**được phân vào nhóm Low-Value Customers.**")
                elif result == 3:
                    st.write("#### **Khách hàng có id**", customer_id, "**được phân vào nhóm Big Spenders.**")
                else:
                    st.write("#### **Khách hàng có id**", customer_id, "**được phân vào nhóm Inactive Customers.**")
                    
            else:
                st.write("#### **Không có khách hàng trong dữ liệu.**")
        else:
            st.write("#### **Thông tin khách hàng không hợp lệ.**")
    else:
        # Nếu người dùng chọn nhập thông tin khách hàng vào dataframe có 3 cột là Recency, Frequency, Monetary
        st.write("##### 2. Thông tin khách hàng")
        # Tạo điều khiển table để người dùng nhập thông tin khách hàng trực tiếp trên table
        st.write("Nhập thông tin khách hàng")
        # Tạo dataframe để người dùng nhập thông tin khách hàng
        df_customer = pd.DataFrame(columns=["Recency", "Frequency", "Monetary"])
        
        # Tạo các slider để nhập giá trị cho cột Recency, Frequency, Monetary
        recency = st.slider("Recency", 1, 1000, 100, key=f"recency_{1}")
        frequency = st.slider("Frequency", 1, 10000, 2000, key=f"frequency_{1}")
        monetary = st.slider("Monetary", 1, 500000, 10000, key=f"monetary_{1}")

        # Cũng có thể thay bằng các điều khiển khác như number_input...
        # Thêm thông tin khách hàng vừa nhập vào dataframe
        df_customer = df_customer.append({"Recency": recency, "Frequency": frequency, "Monetary": monetary}, ignore_index=True)            

        # Get the cluster labels for the new data point
        cluster_labels = loaded_model_1.predict(df_customer)
        if cluster_labels == 0:
            st.write("#### **Khách hàng trênđược phân vào nhóm Regular Customers.**")
        elif cluster_labels == 1:
            st.write("#### **Khách hàng trên được phân vào nhóm Loyal Customers.**")
        elif cluster_labels == 2:
            st.write("#### **Khách hàng trên được phân vào nhóm Low-Value Customers.**")
        elif cluster_labels == 3:
            st.write("#### **Khách hàng trên được phân vào nhóm Big Spenders.**")
        else:
            st.write("#### **Khách hàng trên được phân vào nhóm Inactive Customers.**")


    
    
    
        

        
        

    



