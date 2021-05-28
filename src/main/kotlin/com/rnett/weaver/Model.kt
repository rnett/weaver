package com.rnett.weaver

import org.tensorflow.EagerSession
import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.Session
import org.tensorflow.op.KotlinOps
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32
import org.tensorflow.types.family.TType

abstract class Model(name: String) : Module(name) {
}

annotation class TrainingStep

annotation class Loss

data class WithLoss<T>(val result: T, val loss: Operand<TFloat32>)

class MyModel: Model("MyModel"){

    fun KotlinOps.forward(x: Operand<TFloat32>, y: Operand<TFloat32>): Operand<TFloat32>{
        return math.add(x, y)
    }

    @TrainingStep
    fun KotlinOps.trainStep(x: Operand<TFloat32>, y: Operand<TFloat32>, labels: Operand<TFloat32>): WithLoss<Operand<TFloat32>>{
        val o = forward(x, y)
        val loss = math.sub(o, labels)
        return WithLoss(o, loss)
    }

    //TODO OR.  output is the output of the (only) forward step
    @Loss
    fun KotlinOps.loss(output: Operand<TFloat32>, labels: Operand<TFloat32>): Operand<TFloat32>{
        return math.sub(output, labels)
    }

    // generated

    interface Instance{
        fun <T: TType> T.asOperand(): Operand<T>
        fun forward(x: Operand<TFloat32>, y: Operand<TFloat32>): Operand<TFloat32>
    }

    interface TrainingInstance : Instance {
        fun trainStep(x: Operand<TFloat32>, y: Operand<TFloat32>, labels: Operand<TFloat32>): WithLoss<Operand<TFloat32>>

        // user definable for more complicated training, i.e. GANS.  By default calls the only @TrainStep method.
        fun train(x: Operand<TFloat32>, y: Operand<TFloat32>, labels: Operand<TFloat32>){
            trainStep(x, y, labels)
        }

    }

    inner class GraphInstanceImpl : TrainingInstance {
        private val eagerSession = EagerSession.create()
        private val eagerTf = Ops.create(eagerSession)

        private val graph = Graph()
        private val tf = Ops.create(graph)
        val fowardX = tf.placeholder(TFloat32::class.java)
        val fowardY = tf.placeholder(TFloat32::class.java)
        val forward = forward(fowardX,  fowardY)

        val trainX = tf.placeholder(TFloat32::class.java)
        val trainY = tf.placeholder(TFloat32::class.java)
        val trainLabels = tf.placeholder(TFloat32::class.java)
        val trainStep = trainStep(trainX,  trainY, trainLabels)

        val variables: List<Operand<*>> = listOf()

        val trainStepGradients = graph.addGradients(trainStep.loss.asOutput(), variables.map { it.asOutput() }.toTypedArray())

        //TODO optimizer

        val session = Session(graph)

        override fun <T : TType> T.asOperand(): Operand<T> {
            return eagerTf.constantOf(this)
        }

        override fun trainStep(x: Operand<TFloat32>, y: Operand<TFloat32>, labels: Operand<TFloat32>): WithLoss<Operand<TFloat32>> {
            val output = session.runner()
                .feed(trainX, x.asTensor())
                .feed(trainY, y.asTensor())
                .feed(trainLabels, labels.asTensor())
                .fetch(trainStep.result)
                .fetch(trainStep.loss)
                .apply {
                    trainStepGradients.forEach { fetch(it) }
                }
                .run()
            val result = (output[0] as TFloat32).asOperand()
            val loss = (output[1] as TFloat32).asOperand()
            val gradients = output.drop(2)
            //TODO optimize (part of the graph)?

            return WithLoss(result, loss)
        }

        override fun forward(x: Operand<TFloat32>, y: Operand<TFloat32>): Operand<TFloat32> {
            val output = session.runner()
                .feed(fowardX, x.asTensor())
                .feed(fowardY, y.asTensor())
                .fetch(forward)
                .run()

            return (output[0] as TFloat32).asOperand()
        }

    }

}