1. Chương 1 - Cưỡi ngựa xem hoa
2. Chương 2 - End to End của dự án ML
    1. Tải và kiểm tra dữ liệu thô
    2. Chia train test - Đặc biệt quan tâm kỹ thuật Stratified Sampling
    3. Data Visualize
        - Phân tích tương quan
        - Phát hiện data quirks
    4. Preparing Data
        1. Data Cleaning
            - Dùng hàm Simple Imputor - Để fit - Học giá trị từ Train Set
                + Một số option nâng cao - KNNImputer
                                         - IterativeImputer
            - Sckit-Learn Design
                + Cosistency: Tất cả các đối tượng đều có interface đơng giản
                + Estimators: Các đối tượng có thể học/ước lượng từ Data
                + Transformers: Các estimator vừa học vừa biến đổi
                + Predictors: Một số Estimator có thể dự đoán giá trị mới
                + Ngoài ra nó Pipeline cho phép combine các thể loại Estimators
            - Handling Text and Categorical Attributes
                + Ordinal Encoder: Tạo một cột, mỗi Category mang một unique value  trong cột đó
                + One Hot Encoder: Mỗi Category là một cột, đánh nhãn 1 hoặc 0
            - Feature Scaling và Transformation
                + MinMaxScaler ~ Normalization
                + StandardScaler: Mean = 0 và dùng phân phối chuẩn
                    => Hạn chế sự ảnh hưởng của Outlier
                  - Trong trường hợp dữ liệu thưa thì khỏi lấy data trừ mean -> with_mean = False
                + Trường hợp khó: Heavy Tail
                    Giải pháp:
                    1. Đưa về Log để tính và tuân theo Power Law => Phân phối Gauss
                    2. Dùng phương pháp chia giỏ (Bucketizing)
                + Trường hợp khó: Multimodal Distribution (có nhiều modes, đồ thị phân bố có nhiều đỉnh)
                    1. Chia giỏ -> Chia thành các vùng để học
                    2. Tạo thêm Feature với mõi đỉnh: Mức độ liên quan đến Mode đó
                        - Tính bằng công thức Radio Basis Function (RBF)
                            Trong Sckitlearn có: rbf_kernel()
                        - Gamma: độ khó tính - Tuy xa chút nhưng vẫn giá trị cao
                + Custom Transformer:
                    Dùng FunctionTransfomer(encode_func, inverse_func= decode_func)
                    --> Custom thành Class luôn cũng được
                + Có thể viết thành Pipeline
                    - Pipeline các Transformer
                    - Và có ColumnTransformer --> Cần xác định thêm cột nào sẽ bị ảnh hưởng