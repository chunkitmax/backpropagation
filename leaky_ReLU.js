// translate https://github.com/jgabriellima/backpropagation to js

const math = require('mathjs');

/**
 * return random number within given range
 * @param {number} n start value
 * @param {number} lt end value {Exclusive}
 */
var rand = (n, lt) => ((lt - n) * Math.random() + n);
/**
 * create a new matrix
 * @param {number} i row count
 * @param {number} j column count
 * @param {number} fill [Optional] fill all elements with this value {default: 0}
 */
var newMatrix = (i, j, fill) =>
                {
                    fill = (typeof fill === 'undefined')? 0 : fill;
                    return math.multiply(math.ones(i, j), fill);
                };
/**
 * sigmoid function
 * @param {number} x value
 */
var sigmoid = (x) => ((x > 0.0)? x : 0.01*x);//math.tanh(x);
// var sigmoid = (x) => math.tanh(x);
/**
 * derivative of sigmoid function
 * @param {number} y value
 */
var dsigmoid = (y) => ((y > 0.0)? 1 : 0.01);//(1.0 - math.pow(y, 2));
// var dsigmoid = (y) => (1.0 - math.pow(y, 2));

/**
 * Neural Network class
 * @param {number} inputNum node count of input layer
 * @param {number} hiddenNum node count of hidden layer {assuming only one hidden layer}
 * @param {number} outputNum node count of output layer
 */
var NN =
function (inputNum, hiddenNum, outputNum)
{
    // number of input, hidden, and output nodes
    this.ni = inputNum + 1;
    this.nh = hiddenNum;
    this.no = outputNum;
    
    // activations for nodes
    this.ai = newMatrix(this.ni, 1, 1.0);
    this.ah = newMatrix(this.nh, 1, 1.0);
    this.ao = newMatrix(this.no, 1, 1.0);

    // create weights
    this.wi = newMatrix(this.nh, this.ni);
    this.wo = newMatrix(this.no, this.nh);

    // set them to random vaules
    this.wi = math.map(this.wi, (val) => { return rand(-0.2, 0.2); });
    this.wo = math.map(this.wo, (val) => { return rand(-0.5, 0.5); });

    // last change in weights for momentum
    this.ci = newMatrix(this.nh, this.ni);
    this.co = newMatrix(this.no, this.nh);
};

/**
 * test cases
 * @param {Array} patterns array of input and output pair
 */
NN.prototype.test = 
function (patterns)
{
    console.log();
    console.log('Test results:')
    for (var p of patterns)
        console.log(math.squeeze(p[0]) + "->" + math.squeeze(this.update(p[0])));
};

/**
 * print all weights
 */
NN.prototype.weights = 
function ()
{
    console.log();
    console.log('Input weights:');
    console.log(this.wi._data);
    
    console.log();

    console.log(this.wo._data);
};

/**
 * train the network
 * @param {Array} patterns array of input and output pair
 * @param {number} iterations number of iterations
 * @param {number} N learning rate
 * @param {number} M momentum factor
 */
NN.prototype.train = 
function (patterns, iterations, N, M)
{
    error = Infinity;
    for (var i = 0; math.abs(error) >= 0.0001 && i < iterations; i++)
    {
        error = 0.0;
        for (var p of patterns)
        {
            this.update(p[0]);
            error += this.backPropagate(p[1], N, M);
        }
        // if (i % math.ceil(iterations / 10) == 0)
        if (i % 1000 == 0)
        {
            console.log('error ', error);
            // console.log(this.co._data, this.ci._data);
        }
    }
};

/**
 * update Neural Network
 * @param {Array} inputs input values
 */
NN.prototype.update = 
function (inputs)
{
    if (inputs.length != this.ni - 1)
        throw new Error('input length not match, Expected ' + (this.ni - 1) + ' but ' + inputs.length);
    else if (!Array.isArray(inputs))
        throw new Error('input type not match');
    
    // input activations
    this.ai.subset(math.index([1, this.ni - 1], 0), inputs);

    // hidden activations
    this.ah = math.map(math.multiply(this.wi, this.ai), sigmoid);

    // output activations
    this.ao = math.map(math.multiply(this.wo, this.ah), sigmoid);

    return this.ao;
};

/**
 * do backpropagation
 * @param {Array} targets array of desired output
 * @param {number} N learning rate
 * @param {number} M momentum factor
 */
NN.prototype.backPropagate = 
function (targets, N, M)
{
    if (targets.length != this.no)
        throw new Error('output length not match');

    // transfrom to matrix from array
    targets = math.matrix(targets);
    targets = math.resize(targets, [this.no, 1]);

    // calculate error terms for output
    var outputError = math.subtract(targets, this.ao);
    var dAo = math.map(this.ao, dsigmoid);
    var outputDelta = math.dotMultiply(dAo, outputError);

    // calculate error terms for hidden
    var hiddenError = math.multiply(math.transpose(this.wo), outputDelta);
    var dWo = math.map(this.ah, dsigmoid);
    var hiddenDelta = math.dotMultiply(dWo, hiddenError);

    // update output weights
    var outputChange = math.multiply(outputDelta, math.transpose(this.ah));
    this.wo = math.add(this.wo, math.multiply(N, outputChange));
    this.wo = math.add(this.wo, math.multiply(M, this.co));
    this.co = outputChange;

    // update input weights
    var hiddenChange = math.multiply(hiddenDelta, math.transpose(this.ai));
    this.wi = math.add(this.wi, math.multiply(N, hiddenChange));
    this.wi = math.add(this.wi, math.multiply(M, this.ci));
    this.ci = hiddenChange;

    // calculate error
    errorVector = math.squeeze(outputError);
    var error = 0.0;
    if (typeof errorVector === 'number')
        error = 0.5 * math.pow(errorVector, 2);
    else
        error = 0.5 * math.dot(errorVector, errorVector);
    
    return error;
};

/****************************** */

var pat = [ [[0,0], [1]],
            [[0,1], [0]],
            [[1,0], [1]],
            [[1,1], [0]] ];

n = new NN(2, 4, 1);
n.train(pat, 50000, 0.001, 0.005);
n.test(pat);
// n.weights();

/***************************** */

console.log('\nEnd');
