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
4. Chương 4: Chương này nhiều công thức quá, viết vô tập nha ae
    1. Linear Regression
        1. The Normal Equation
        2. Computational Complexity
    2. Gradient Descent
        1. Batch Gradient Descent
        2. Stochastic Gradient Descent
        3. Mini-batch Gradient Descent
    3. Polynominal Regression
    4. Learning Curves
    5. Regularized Linear Models
        1. Ridge Regression
            - w^2: Giữ tất cả feature, chỉ làm chậm, không đưa về 0
        2. Lasso Regression
            - w: Có thể đưa nhiều feature về 0
        3. Elastic Net Regression
            - Kết hợp cả 2
            - Dùng khi
                + Khi số Feature lớn
                + Có nhiều Feature liên quan
                + Khi lasso chọn một feature và loại các feature tương tự
        4. Early Stopping
    5. Logistic Regression
        1. Estimating Probabilities
            - Logistic Function: Output chỉ cần là số thuộc [0,1]
        2. Training and Cost Function
            - Loss Function: Mỗi lần cập nhật là giảm log(nhãn đúng)
        3. Decision Boundaries
            - Vị trí mà tỉ lệ nhận diện là 50 50
        4. Softmax Regression
            - multinomial logistic regression
            - Là công thức tổng quát của Logistic Regression để xử lý trên đa class
        5. Cross Entropy (Loss Function trong Classification)
5. Chương 5: Support Vector Machine
    1. Linear SVM Classification
        - SVM Chọn đường phân cách sao cho xa các điểm dữ liệu gần nhất có thể
        - Margin: Hai đường nét đứt song song || Khoảng cách từ đường phân lớp đến các điểm gần nhất
        - Hard Margin: Không cho phép bất kỳ điểm dữ liệu nằm ngoài vùng margin
        - Soft Margin: Cho phép một số điểm dữ liệu
        - C paramater: Cho phép nhưng sẽ có hệ số phạt
                        C càng cao phạt càng nặng nếu nằm ngoài vùng
                    - C cao -> Dễ Underfitting
                    - C thấp -> Dễ Overfitting
        * Mấy cái điểm ngoại lệ đó là một Support Vector
    2. Non Linear SVM Classification
        1. Polynominal Kernel
        2. Trong SVC nó có paramater (kernel="poly", degree=3) mô hình nó tự dựng thành hàm đa thức mà không cần thật sự tạo thêm đặc trưng
            -> Cái này người ta làm sẵn nên sẽ tối ưu hơn.
        3. Similarity Features
            - Đây là một cách khác xử lý Non Linear Data
            - Ta sẽ thêm Feature là độ liên quan đến dữ liệu rbf_kernel đối với một landmark nào đó.
            - Vậy làm sao để biết điểm nào là Landmark quan trọng để mà chọn?
            ==> Chọn hết cũng được mà tăng chi phí tính toán
            - Hàm tính toán độ tương đồng: Gaussian RBF Kernel
            - Kernel trick mới làm được chỉnh paramater kernel="rbf", gamma=5, C=0.001
            - Kernel Trick tự cho ra performance good mà không thêm nhiều điểm dữ liệu
        4. SVM Linear Regression
            - Chỉ cần thay đổi hàm mục tiêu, ta sẽ cố gắng sao cho tất cả các điểm nằm trong Margin
            --> Epsilon = khoảng cách 2 margin, epsilon càng nhỏ
            --> Càng nhiều support vector -> Overfitting
        5. Under the hoot of SVM Linear Regression
            - Skip toán nha ae <(")
6. Chương 6: Decisions Tree

7. Chương 7: Ensemble Learning and Random Forest

8. Chương 8: Dimensionally Reduction

9. Chương 9: Unsupervised Learning Techniques

10. Chương 10: Introduction to Artificle Neural Network with Keras

11. Training Deep Neural Network

12. Custom Models and Training with Tensorflow

13. Loading and Preprocessing Data with Tensorflow

14. Deep Computer Vision Using Convolutional Neural Networks

15. Processing Sequences Using RNNs and CNNs

16. Natural Language Processing with RNNs and Attention

17. Autoencoders, GANs, and Diffusion Models

18. Reinforcement Learning

19. Training and Deploying Tensorflow Models at Scale

