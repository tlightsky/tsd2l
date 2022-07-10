import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis';
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Variable} from "@tensorflow/tfjs"
import {Timer} from "timer-node"

(async function () {
    const timer = new Timer({label: 'test-timer'})
    timer.start()

    let outText: string = ''

    let n: number = 10000
    let a = tf.ones([n])
    let b = tf.ones([n])
    let c = tf.variable(tf.zeros([n]))
    c.assign(a.add(b))

    outText += `time1: ${timer.format('%label [%s] seconds [%ms] ms')}\n`
    let d: number;
    for (let i: number = 0; i < n; i++) {
        d = a.dataSync()[i]+b.dataSync()[i]
        // console.log(d)
    }
    outText += `time2: ${timer.format('%label [%s] seconds [%ms] ms')}\n`

    // def normal(x, mu, sigma):
    // p = 1 / math.sqrt(2 * math.pi * sigma**2)
    // return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

    // def synthetic_data(w, b, num_examples):  #@save
    // """生成 y = Xw + b + 噪声。"""
    // X = tf.zeros((num_examples, w.shape[0]))
    // X += tf.random.normal(shape=X.shape)
    // y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
    // y += tf.random.normal(shape=y.shape, stddev=0.01)
    // y = tf.reshape(y, (-1, 1))
    // return X, y

    let syntheticData = function (w: Tensor, b: number, numExamples: number) {
        let X: Tensor = tf.zeros([numExamples, w.shape[0]])
        X = X.add(tf.randomNormal(X.shape))
        let y: Tensor = X.matMul(w.reshape([-1, 1])).add(b)
        y = y.add(tf.randomNormal(y.shape, 0, 0.01))
        y = y.reshape([-1, 1])
        return [X, y]
    }
    let [X, y] = syntheticData(tf.tensor([2, -3.4]), 4.2, 1000)
    // X.print(true)
    // y.print(true)
    console.log(X.arraySync())
    console.log(y.arraySync())
    let XArr = X.arraySync() as number[][]
    let yArr = y.arraySync() as number[][]

    await tfvis.render.scatterplot({
        name: 'Synthetic Data'
    }, {
        values: XArr.map((x, i) => {
            return {x: x[1], y: yArr[i][0]}
        })
       // series: y.dataSync(),
    })
    // console.log(X)
    // console.log(y)

    let outEle = document.getElementById('add')
    if (outEle !== null) {
        outEle.innerText = outText
    }
})();
