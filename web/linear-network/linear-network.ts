import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis';
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Variable} from "@tensorflow/tfjs"
import {Timer} from "timer-node"

function shuffle(array: number[]) {
    for (let i = array.length - 1; i > 0; i--) {
        let j = Math.floor(Math.random() * (i + 1)); // random index from 0 to i

        // swap elements array[i] and array[j]
        // we use "destructuring assignment" syntax to achieve that
        // you'll find more details about that syntax in later chapters
        // same can be written as:
        // let t = array[i]; array[i] = array[j]; array[j] = t
        [array[i], array[j]] = [array[j], array[i]];
    }
}

(async function () {
    let outText: string = '';

    (() => {
        const timer = new Timer({label: 'test-timer'})
        timer.start()


        let n: number = 10000
        let a = tf.ones([n])
        let b = tf.ones([n])
        let c = tf.variable(tf.zeros([n]))
        c.assign(a.add(b))

        outText += `time1: ${timer.format('%label [%s] seconds [%ms] ms')}\n`
        let d: number;
        for (let i: number = 0; i < n; i++) {
            d = a.dataSync()[i] + b.dataSync()[i]
            // console.log(d)
        }
        outText += `time2: ${timer.format('%label [%s] seconds [%ms] ms')}\n`
    })()

    // def normal(x, mu, sigma):
    // p = 1 / math.sqrt(2 * math.pi * sigma**2)
    // return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

    // def synthetic_data(w, b, num_examples):  #@save
    // """生成 labels = Xw + b + 噪声。"""
    // features = tf.zeros((num_examples, w.shape[0]))
    // features += tf.random.normal(shape=features.shape)
    // labels = tf.matmul(features, tf.reshape(w, (-1, 1))) + b
    // labels += tf.random.normal(shape=labels.shape, stddev=0.01)
    // labels = tf.reshape(labels, (-1, 1))
    // return features, labels

    let syntheticData = function (w: Tensor, b: number, numExamples: number) {
        let X: Tensor = tf.zeros([numExamples, w.shape[0]])
        X = X.add(tf.randomNormal(X.shape))
        let y: Tensor = X.matMul(w.reshape([-1, 1])).add(b)
        y = y.add(tf.randomNormal(y.shape, 0, 0.01))
        y = y.reshape([-1, 1])
        return [X, y]
    }
    let [features, labels] = syntheticData(tf.tensor([2, -3.4]), 4.2, 1000)
    // features.print(true)
    // labels.print(true)
    // console.log(features.arraySync())
    // console.log(labels.arraySync())
    let featuresArr = features.arraySync() as number[][]
    let labelsArr = labels.arraySync() as number[][]

    await tfvis.render.scatterplot({
        name: 'Synthetic Data Y'
    }, {
        values: featuresArr.map((x, i) => {
            return {x: x[0], y: labelsArr[i][0]}
        })
    })

    await tfvis.render.scatterplot({
        name: 'Synthetic Data Z'
    }, {
        values: featuresArr.map((x, i) => {
            return {x: x[1], y: labelsArr[i][0]}
        })
    })

//     def data_iter(batch_size, features, labels):
    //     num_examples = len(features)
    //     indices = list(range(num_examples))
    // # 这些样本是随机读取的，没有特定的顺序
    //     random.shuffle(indices)
    //     for i in range(0, num_examples, batch_size):
    //     j = tf.constant(indices[i: min(i + batch_size, num_examples)])
    //     yield tf.gather(features, j), tf.gather(labels, j)

    let dataIter = function* (batchSize: number, features: Tensor, labels: Tensor) {
        let exampleNumber: number = labels.size
        let indices = [...Array(exampleNumber).keys()]
        shuffle(indices)
        for (let i: number = 0; i < exampleNumber; i += batchSize) {
            // yield indices[i]
            let sliced = indices.slice(i, Math.min(i + batchSize, exampleNumber))
            let j: Tensor = tf.tensor(sliced, [sliced.length], 'int32')
            yield [features.gather(j), labels.gather(j)]
        }
    }
    let batchSize = 10;
    let feature;
    let label;
    // for ([feature, label] of dataIter(batchSize, features, labels)) {
    //     feature.print()
    //     label.print()
    // }

    let w = tf.variable(tf.randomNormal([2, 1], 0, 0.01), true)
    let b = tf.variable(tf.zeros([1]), true)

    function linreg(X: Tensor, w: Tensor, b: Tensor) {
        return X.matMul(w).add(b)
    }

    // def squared_loss(y_hat, y):  #@save
    // """均方损失。"""
    //     return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2
    function squaredLoss(y_hat: Tensor, y: Tensor) {
        return y_hat.sub(y).square().mean()
    }

    function sgd(params: Variable[], grads: Tensor[], learningRate: number, batch_size: number) {
        for (let i = 0; i < params.length; i++) {
            params[0].assign(params[0].sub(grads[0].mul(learningRate).div(batch_size)))
            params[1].assign(params[1].sub(grads[1].mul(learningRate).div(batch_size)))
        }
    }

    // tf.train.sgd(0.1).minimize()
    let lr = 0.03,
        numEpochs = 3,
        net = linreg,
        loss = squaredLoss;
//     for epoch in range(num_epochs):
    //     for X, y in data_iter(batch_size, features, labels):
    //     with tf.GradientTape() as g:
        //     l = loss(net(X, w, b), y)  # `X`和`y`的小批量损失
        // # 计算l关于[`w`, `b`]的梯度
    //     dw, db = g.gradient(l, [w, b])
    // # 使用参数的梯度更新参数
    //     sgd([w, b], [dw, db], lr, batch_size)
    //     train_l = loss(net(features, w, b), labels)
    //     print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')
    for (let i = 0; i < numEpochs; i++) {
        for(let [X, y] of dataIter(batchSize, features, labels)) {
            let l = (w: Tensor, b: Tensor) => loss(net(X, w, b), y)
            let g = tf.grads(l)
            let [dw, db] = g([w, b])
            // dw.print()
            // db.print()
            // console.log('Start------')
            // l(w, b).print()
            sgd([w, b], [dw, db], lr, batchSize)
            // l(w, b).print()
            // console.log('End------')
        }
        let train_l = loss(net(features, w, b), labels)
        // print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')
        console.log(`epoch ${i + 1}, loss ${train_l}`)
    }


    let outEle = document.getElementById('add')
    if (outEle !== null) {
        outEle.innerText = outText
    }
})();
