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
    5. Select and Train a Model
        1. Train model thì giới thiệu chill chill LinearRegression, DecisionTreeRegressor
        2. Dùng Cross Validation
            - trong Scikit-Learn nó là cross_val_score
    6. Fine-Tune your model
        1. Grid Search
        2. Randomized Search
        3. HalvingGridSearch
        4. Gộp nhiều mô hình (Essembled Methods)
        5. Phân tích mô hình tốt nhất và Lỗi của chúng
        6. Evaluate
    7. Launch and Monitoring
3. Chương 3: Classification
    1. MNIST
    2. Binary Classifier
    3. Performance Measures
        1. Vẫn dùng KFold
            -  Nhưng chỉ KFold là chưa đủ
            - Vì nó lệch nhãn -> Dự đoán nhãn nào bị lệch thì accuracy tự động cao
        2. F1-Score
            - Precision: Hàng ngang 1 - Tỷ lệ dự đoán đúng thật sự đúng
            - Recall: Hàng dọc 1 - Tỷ lệ vét được tất cả trường hợp đúng
            - F1 Score:  Tích trên trung bình
                     = Recall * Precision * 2 / (Recall + Precision)
        3. ROC và AUC
            + ROC = Recall / Fall-out
            + Recall: Tỷ lệ tỉnh táo vết được tập đúng
            + Fall-out: Tỷ lệ nhầm lẫn trên tập sai
            + AUC: Area Under Curve, vùng diện tích dưới đường cong càng to mô hình càng tốt
        4.  Dùng F1 Score khi quan tâm đến nhãn đúng, hay nhãn đó nó bị lệch (ít quá cần quan tâm nhiều hơn)
            -> Chỗ này nói hơi nhầm xíu, thật ra nên check cái đồ thị Precision Recall thay vì ROC
    4. Multi-class Classification
        - Ý tưởng ngây thơ 1: One versus Rest (OvR hoặc OvA): Mỗi class một mô hình, xong tính điểm các mô hình (Nó có phải lớp A hay không ?)
        - Ý tưởng ngây thơ 2: One versus One (OvO): Ý tưởng na ná nhưng mà sẽ xây mô hình phân loại A hoặc B
    5. Error Analysis
        - Bắt đầu in ra Confussion matrix để coi các nhãn nào bị lẫn lộn nhiều nhất
        - Số to quá thì hiện phần trăm nhầm lẫn
    6. MultiLabel Classification
        1. Nếu mô hình ML không hỗ trợ thì dùng ClassifierChain
            -  Mỗi mô hình trong ClassifierChain phụ trách một label cần xử lý và mốc nối với nhau
        2. Một số mô hình ML giải quyết được Multilabel Classification
    7. Multioutput Classification
        - Chương này thì cũng không nói gì mấy
        - Multiouput vì nó vừa phải học nếu đó là Số 7 thì cần xóa noise chỗ nào, giữ color chỗ nào
        - Cách giải thì cũng chỉ ccaafn thay phần y cần predict thành X_train lúc Clean background là xong
