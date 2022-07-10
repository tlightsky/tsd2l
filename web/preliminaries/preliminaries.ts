import * as tf from '@tensorflow/tfjs'
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Variable} from "@tensorflow/tfjs"

(function () {
    let x: Tensor1D = tf.range(0, 12)

    let outText: string = ''
    outText += `\nrange(0,12):\n`
    outText += `${x}\n`
    outText += `\nshape:\n`
    outText += `${x.shape}\n`

    outText += `\nreshape([3,4]):\n`
    outText += `${x.reshape([3, 4])}\n`
    outText += `\nx.reshape([3, 4]).shape:\n`
    outText += `${x.reshape([3, 4]).shape}\n`

    outText += `\ntf.zeros([2, 3, 4]):\n`
    outText += `${tf.zeros([2, 3, 4])}\n`

    outText += `\ntf.ones([2, 3, 4]):\n`
    outText += `${tf.ones([2, 3, 4])}\n`

    outText += `\n${tf.randomNormal([3, 4])}:\n`
    outText += `${tf.randomNormal([3, 4])}\n`

    outText += `\ntf.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]):\n`
    outText += `${tf.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])}\n`

    outText += `\ntf.add([1.0, 2, 4, 8], [2.0, 2, 2, 2]):\n`
    outText += `${tf.add([1.0, 2, 4, 8], [2.0, 2, 2, 2])}\n`

    outText += `\ntf.sub([1.0, 2, 4, 8], [2.0, 2, 2, 2]):\n`
    outText += `${tf.sub([1.0, 2, 4, 8], [2.0, 2, 2, 2])}\n`

    outText += `\ntf.mul([1.0, 2, 4, 8], [2.0, 2, 2, 2]):\n`
    outText += `${tf.mul([1.0, 2, 4, 8], [2.0, 2, 2, 2])}\n`

    outText += `\ntf.div([1.0, 2, 4, 8], [2.0, 2, 2, 2]):\n`
    outText += `${tf.div([1.0, 2, 4, 8], [2.0, 2, 2, 2])}\n`

    outText += `\ntf.**([1.0, 2, 4, 8], [2.0, 2, 2, 2]):\n`
    outText += `${'???'}\n`

    outText += `\ntf.exp([1.0, 2, 4, 8]):\n`
    outText += `${tf.exp([1.0, 2, 4, 8])}\n`

    let X: Tensor1D = tf.range(0, 12, 1, 'float32').reshape([3, 4])
    let Y: Tensor1D = tf.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    outText += `\nlet X: Tensor1D = tf.range(0, 12, 1, 'float32').reshape([3, 4]):\n`
    outText += `${tf.range(0, 12, 1, 'float32').reshape([3, 4])}\n`
    outText += `\nlet Y: Tensor1D = tf.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]):\n`
    outText += `${tf.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])}\n`
    outText += `\ntf.concat([X, Y], 0), tf.concat([X, Y], 1):\n`
    outText += `${tf.concat([X, Y], 0)}, \n${tf.concat([X, Y], 1)}\n`
    outText += `\ntf.equal(X, Y):\n`
    outText += `${tf.equal(X, Y)}\n`

    outText += `\nX.sum():\n`
    outText += `${X.sum()}\n`

    let a: Tensor1D = tf.range(0, 3).reshape([3, 1])
    let b: Tensor1D = tf.range(0, 2).reshape([1, 2])
    outText += `\nlet a: Tensor1D = tf.range(0, 3).reshape([3, 1]):\n`
    outText += `${a}\n`
    outText += `\nlet b: Tensor1D = tf.range(0, 2).reshape([1, 2]):\n`
    outText += `${b}\n`

    outText += `\na.add(b):\n`
    outText += `${a.add(b)}\n`

    // outText += `${X}\n`
    outText += `\nX.slice(1, 3):\n`
    outText += `${'???'}\n`

    let XVar: Variable = tf.variable(X)
    outText += `\nlet XVar: Variable = tf.variable(X):\n`
    outText += `${XVar}\n`

    outText += `\nX.slice(1, 3).assign(9):\n`
    outText += `${'???'}\n`

    outText += `@tf.function? ${'???'}\n`

    let outEle = document.getElementById('ndarray')
    if (outEle !== null) {
        outEle.innerText = outText
    }
})();

(async function () {
    let x: Tensor1D = tf.range(0, 4)
    let outText: string = ''

    outText += `\nlet x: Tensor1D = tf.range(0, 4):\n`
    outText += `${x}\n`

    outText += `\n(await x.data())[3]):\n`
    outText += `${(await x.data())[3]}\n`

    outText += `\nx.size:\n`
    outText += `${x.size}\n`

    outText += `\nx.shape:\n`
    outText += `${x.shape}\n`

    let A: Tensor2D = tf.range(0, 20).reshape([5, 4])
    outText += `\nlet A: Tensor2D= tf.range(0, 20).reshape([5, 4]):\n`
    outText += `${A}\n`
    outText += `\nA.transpose():\n`
    outText += `${A.transpose()}\n`

    let B: Tensor2D = tf.tensor2d([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
    outText += `\nlet B: Tensor2D = tf.tensor2d([[1, 2, 3], [2, 0, 4], [3, 4, 5]]):\n`
    outText += `${B}\n`
    outText += `\nB.equal(B.transpose()):\n`
    outText += `${B.equal(B.transpose())}\n`

    let X: Tensor3D = tf.range(0, 24).reshape([2, 3, 4])
    outText += `\nlet X: Tensor3D = tf.range(0, 24).reshape([2, 3, 4]):\n`
    outText += `${X}\n`

    outText += `\nA:\n`
    outText += `${A}\n`
    outText += `\nA.add(A):\n`
    outText += `${A.add(A)}\n`
    outText += `\nA.mul(A):\n`
    outText += `${A.mul(A)}\n`

    outText += `\nX.add(2):\n`
    outText += `${X.add(2)}\n`
    outText += `\nX.mul(2):\n`
    outText += `${X.mul(2)}\n`

    outText += `\nA.sum():\n`
    outText += `${A.sum()}\n`
    outText += `\nA.sum(0):\n`
    outText += `${A.sum(0)}\n`
    outText += `\nA.sum(1):\n`
    outText += `${A.sum(1)}\n`
    outText += `\nA.sum([0, 1]):\n`
    outText += `${A.sum([0, 1])}\n`
    outText += `\nA.mean(0):\n`
    outText += `${A.mean(0)}\n`
    outText += `\nA.sum(1, true):\n`
    outText += `${A.sum(1, true)}\n`
    outText += `\nA.div(A.sum(1, true)):\n`
    outText += `${A.div(A.sum(1, true))}\n`

    outText += `\nA.cumsum(0):\n`
    outText += `${A.cumsum(0)}\n`

    let y: Tensor1D = tf.ones([4], 'float32')
    outText += `\nlet y = tf.ones([4], 'float32'):\n`
    outText += `${y}\n`
    outText += `\nx.dot(y):\n`
    outText += `${x.dot(y)}\n`

    outText += `\nx.mul(y).sum():\n`
    outText += `${x.mul(y).sum()}\n`

    outText += `\ntf.linalg.matvec(A, x):\n`
    outText += `${'???'}\n`

    let B2 = tf.ones([4, 3], 'float32')
    outText += `\nA.matMul(B2):\n`
    outText += `${A.matMul(B2)}\n`

    let u: Tensor1D = tf.tensor1d([3.0, -4.0])
    outText += `\nlet u: Tensor1D = tf.tensor1d([3.0, -4.0]):\n`
    outText += `${u}\n`
    outText += `\nu.norm():\n`
    outText += `${u.norm()}\n`
    outText += `\ntf.ones([4, 9]).norm():\n`
    outText += `${tf.ones([4, 9]).norm()}\n`

    outText += `\nX.size:\n`
    outText += `${X.size}\n`


    let outEle = document.getElementById('linear-algebra')
    if (outEle !== null) {
        outEle.innerText = outText
    }
})();


(async function () {
    let outText: string = ''

    const f = function (x: Tensor) {
        return x.square()
    }
    const g = tf.grad(f)
    const x = tf.tensor1d([2, 3]);

    outText += `\ng(x):\n`
    outText += `${g(x)}\n`

    let outEle = document.getElementById('grad')
    if (outEle !== null) {
        outEle.innerText = outText
    }
})();