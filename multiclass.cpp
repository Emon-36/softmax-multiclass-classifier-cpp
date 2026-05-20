#include <iostream>
#include <cmath>
#include <algorithm>
using namespace std;

float x[4][4] = {
    {2.44, 0.34, 5.67, 0.34},
    {12.3, 4.34, 0.12, 0.31},
    {0.44, 10.34, 5.67, 0.34},
    {0.12, 0.14, 5.67, 9.34}  
};

float test[4][4] = {
    {3.01, 0.12, 6.23, 0.23},
    {9.03, 8.34, 0.09, 0.10},
    {0.44, 11.34, 8.67, 0.30},
    {0.14, 0.34, 5.67, 6.34}
};

void softmax(float input[4], float output[4]){
    float max_value = max({input[0], input[1], input[2], input[3]});
    float sum = 0.0;
    for (size_t i = 0; i < 4; i++) {
        output[i] = exp(input[i] - max_value);
        sum += output[i];
    }
    for (int i = 0; i < 4; i++) {
        output[i] /= sum;
    }
}

void weight_avg(float accum[4][4], float w[4][4], float lr) {
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            w[i][j] -= lr * (accum[i][j] / 4.0f);
        }
    }
}

void bias_avg(float accum[4], float bias[4], float lr) {
    for (size_t i = 0; i < 4; i++) {
        bias[i] -= lr * (accum[i] / 4.0f);
    }
}

int main() {
    float w[4][4] = {
        {0.1, -0.2, 0.05, 0.1},
        {-0.1, 0.3, -0.15, 0.2},
        {0.05, 0.01, 0.2, -0.1},
        {0.15, -0.1, 0.1, 0.25}
    };
    
    float lr = 0.05;
    float bias[4] = {2.0, 1.23, -5.67, 2.12};
    float y_actual[4][4] = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    };

    for (size_t epoch = 1; epoch < 1000; epoch++) {    
        float epoch_loss = 0;
        float probability[4] = {0};
        
        float weight_grad[4][4] = {0};
        float bias_grad[4] = {0};

        for (size_t s = 0; s < 4; s++) {
            float logits[4] = {0};
            float gradient[4] = {0};
            
    
            for (size_t i = 0; i < 4; i++) {
                float sum = 0;
                for (size_t j = 0; j < 4; j++) {
                    sum += (w[i][j] * x[s][j]);
                }
                logits[i] = sum + bias[i];
            }
            
            softmax(logits, probability);
            
        
            for (size_t i = 0; i < 4; i++) {
                if (y_actual[s][i] == 1.0f) {
                    epoch_loss -= log(probability[i] + 1e-7f);
                }
            }
            
            
            for (size_t i = 0; i < 4; i++) {
                gradient[i] = probability[i] - y_actual[s][i]; // dZ
                
                
                bias_grad[i] += gradient[i]; 
                
                
                for (size_t j = 0; j < 4; j++) {
                    weight_grad[i][j] += gradient[i] * x[s][j]; 
                }
            }
        }

       
        weight_avg(weight_grad, w, lr);
        bias_avg(bias_grad, bias, lr);
        
       
        if (epoch % 100 == 0 || epoch == 1) {
            cout << "Epoch " << epoch 
                 << "  Last Sample Pred: [" << probability[0] << ", " << probability[1] << ", " << probability[2] << ", " << probability[3] << "]"
                 << "  Avg Loss: " << (epoch_loss / 4.0f) << endl;
        }
    }
    cout<<endl<<endl;
   for(int s=0;s<4;s++){	
	   float logits[4] = {0};
    float test_prob[4] = {0};
 
    for (size_t i = 0; i < 4; i++) {
        float sum = 0;
        for (size_t j = 0; j < 4; j++) {
            sum += (w[i][j] * test[s][j]);
        }
        logits[i] = sum + bias[i];
    }
    
    softmax(logits, test_prob);
    
    cout << "Test Sample " << 1 << " Pred: [" 
         << test_prob[0] << ", " << test_prob[1] << ", " 
         << test_prob[2] << ", " << test_prob[3] << "]" << endl;
 for (size_t i = 0; i < 4; i++)
    {
        if (test_prob[i]>0.6)
        {
            cout<<"Class "<<i<<endl;
            if (i==0)
            {
                cout<< "Data pattern : [HIGH, LOW, HIGH, LOW]";
            }

            else if (i==1)
            {
                cout<< "Data pattern : [HIGH, HIGH, LOW, LOW]";
            }
            else if (i==2)
            {
                cout<< "Data pattern : [LOW, HIGH, HIGH, LOW]";
            }
            else if (i==3)
            {
                cout<< "Data pattern : [LOW, LOW, HIGH, HIGH]";
            }
        }
        
    }


    
    

    return 0;
}