package com.example.irisprediction

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.irisprediction.ml.Iris
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        var bt : Button = findViewById(R.id.button);
        bt.setOnClickListener(View.OnClickListener {
            var ed1 : EditText = findViewById(R.id.editTextNumberDecimal);
            var ed2 : EditText = findViewById(R.id.editTextNumberDecimal2);
            var ed3 : EditText = findViewById(R.id.editTextNumberDecimal3);
            var ed4 : EditText = findViewById(R.id.editTextNumberDecimal4);
            var tv : TextView = findViewById(R.id.textView);

            var v1=0.0f;
            var v2=0.0f;
            var v3=0.0f;
            var v4=0.0f;

            try{
                v1 = ed1.text.toString().toFloat();
                v2 = ed2.text.toString().toFloat();
                v3 = ed3.text.toString().toFloat();
                v4 = ed4.text.toString().toFloat();
            }catch(e : Exception){
                tv.setText("Enter Some Input")
                return@OnClickListener
            }


            if(v1<=0 || v2<=0 || v3<=0 || v4<=0 ){
                tv.setText("Input Cannot be 0 or less than 0")
                return@OnClickListener
            }

            val model = Iris.newInstance(this)

            var byteBuffer : ByteBuffer = ByteBuffer.allocateDirect(4*4);
            byteBuffer.putFloat(v1);
            byteBuffer.putFloat(v2);
            byteBuffer.putFloat(v3);
            byteBuffer.putFloat(v4);

            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 4), DataType.FLOAT32)
            inputFeature0.loadBuffer(byteBuffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            tv.setText("Iris-setosa : "+outputFeature0[0].toString() + "\nIris-versicolor : "
                    +outputFeature0[1].toString() + "\nIris-virginica : "
                    +outputFeature0[2].toString()  )

            // Releases model resources if no longer used.
            model.close()
        })



    }
}