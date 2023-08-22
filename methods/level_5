using System;
using System.Threading.Tasks;

public class MatrixMultiplier
{
    public double[,] Multiply(double[,] matrixA, double[,] matrixB)
    {
        if (matrixA.GetLength(1) != matrixB.GetLength(0))
            throw new InvalidOperationException("Matrices cannot be multiplied");

        double[,] result = new double[matrixA.GetLength(0), matrixB.GetLength(1)];

        Parallel.For(0, result.GetLength(0), i =>
        {
            for (int j = 0; j < result.GetLength(1); j++)
            {
                double sum = 0;
                for (int k = 0; k < matrixA.GetLength(1); k++)
                {
                    sum += matrixA[i, k] * matrixB[k, j];
                }
                result[i, j] = sum;
            }
        });

        return result;
    }
}
