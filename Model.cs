using System;
using System.Numerics;
using System.Security.Claims;

namespace LinearRegression
{
    public class Model
    {
        public float Weight { get; set; }
        public float Bias { get; set; }
        public float LearningRate { get; set; }

        private List<Vector2> SampleData;

        public Model(List<Vector2> sampleData, float learningRate) 
        {
            this.Weight = 0.0f;
            this.Bias = 0.0f;

            this.SampleData = sampleData;
            this.LearningRate = learningRate;
        }

        public void Train(int epoch)
        {
            for (int i = 0; i < epoch; i++)
            {
                float grad_weight = 0;
                float grad_bias = 0;

                foreach (Vector2 sample in SampleData)
                {
                    grad_weight += -(2.0f / SampleData.Count) * sample.X * (sample.Y - Calc(sample.X));
                    grad_bias += -(2.0f / SampleData.Count) * (sample.Y - Calc(sample.X));
                }

                Weight = Weight - LearningRate * grad_weight;
                Bias = Bias - LearningRate * grad_bias;
            }
        }

        private float Calc(float x)
        {
            return Weight * x + Bias;
        }

        public Vector2 Predict(float x)
        {
            return new Vector2(x, Calc(x));
        }
    }
}
