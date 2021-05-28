package com.rnett.weaver

import org.tensorflow.Operand
import org.tensorflow.op.KotlinOps
import org.tensorflow.types.TFloat32
import kotlin.contracts.InvocationKind
import kotlin.contracts.contract
import kotlin.properties.ReadOnlyProperty
import kotlin.reflect.KProperty

class HasntBuiltYetException(val key: String) :
    IllegalStateException("Can't get built memory \"$key\" before the module is built (i.e. called for the first time)")

//TODO doesn't work super well for late-init (i.e. input dependent) properties.  remember{} works fine but is hard to key.  Is it actually fine?
//  lazy + initWith?  don't like, hard to typecheck
//  Possibly have lazyMemory as an object, remember(LazyMemory){} and LazyMemory.get()?
//
abstract class Module(open val name: String) {

    protected annotation class Forward

    @PublishedApi
    internal val memories = mutableMapOf<String, Any?>()

    inline fun <T> remember(key: String, value: (String) -> T): T {
        contract { callsInPlace(value, InvocationKind.AT_MOST_ONCE) }
        return memories.getOrPut(key) { value(key) } as T
    }

    inline fun <T> remember(value: (String) -> T): T {
        contract { callsInPlace(value, InvocationKind.AT_MOST_ONCE) }
        error("Compiler should substitute this out")
    }

    abstract inner class Memory<T> @PublishedApi internal constructor() {
        protected abstract fun initial(): T
        inline operator fun getValue(thisRef: Any?, property: KProperty<*>): T {
            return remember(property.name) { `access$initial`() }
        }

        @PublishedApi
        internal fun `access$initial`() = initial()
    }

    inline fun <T> memory(crossinline initial: () -> T) = object : Memory<T>() {
        override fun initial(): T = initial()
    }

    @PublishedApi
    internal fun <T> setupBuildMemory(key: String, initial: KotlinOps.() -> T) {
        onInit {
            memories[key] = initial()
        }
    }

    inner class BuildMemoryDelegate<T> @PublishedApi internal constructor(val initial: KotlinOps.() -> T) {
        operator fun provideDelegate(thisRef: Any?, property: KProperty<*>): ReadOnlyProperty<Any?, T> {
            val key = property.name
            setupBuildMemory(key, initial)
            return ReadOnlyProperty<Any?, T> { _, _ ->
                remember(key) { throw HasntBuiltYetException(it) }
            }
        }
    }

    inline fun <T> buildMemory(noinline initial: KotlinOps.() -> T) = BuildMemoryDelegate(initial)

    @PublishedApi
    internal val initializers = mutableListOf<KotlinOps.() -> Unit>()

    open fun KotlinOps.onInit() {

    }

    protected fun onInit(block: KotlinOps.() -> Unit) {
        initializers.add(block)
    }

    @PublishedApi
    internal var hasInited = false

    protected inline fun doInit(tf: KotlinOps) {
        if (hasInited) return
        tf.onInit()
        initializers.forEach { tf.it() }
        hasInited = true
    }
}

/**
 * [activation] should be framework Activation once it's a functional interface.
 *
 * TODO receiver order is wrong, but need decorators to fix
 */
class MyDenseLayer(val n: Int, val activation: KotlinOps.(Operand<TFloat32>) -> Operand<TFloat32>) : Module("DenseLayer") {

    val b by buildMemory { variable(ones(array(n), TFloat32::class.java)) }

    @Forward
    fun KotlinOps.call(x: Operand<TFloat32>): Operand<TFloat32> {
        val inputDims = x.shape().size(1)
        val W = remember { variable(ones(array(inputDims.toInt(), n), TFloat32::class.java)) }
        return activation(math.add(linalg.matMul(x, W), b))
    }

    // generated
    operator fun KotlinOps.invoke(x: Operand<TFloat32>): Operand<TFloat32> =
        withSubScope(name)
            .also { doInit(it) }
            .call(x)
}