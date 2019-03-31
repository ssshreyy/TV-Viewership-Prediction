import { Injectable } from '@angular/core';

@Injectable()
export class VisualsService {
    xAxisData = [];
    data1 = [];
    data2 = [];
    constructor() {
        for (var i = 0; i < 100; i++) {
            this.xAxisData.push('Type ' + i);
            this.data1.push((Math.sin(i / 5) * (i / 5 - 10) + i / 6) * 5);
            this.data2.push((Math.cos(i / 5) * (i / 5 - 10) + i / 6) * 5);
        }
    }

    BarOption;
    PieOption;
    LineOption;
    AnimationBarOption;
    ScatterOption;


    getGradientOption(x,y) {
        return {
            visualMap: [{
                show: false,
                type: 'continuous',
                seriesIndex: 0,
                min: 8,
                max: 4
            }],
        
            tooltip: {
                trigger: 'axis'
            },
            xAxis: [{
                data: x
            }, {
                data: x,
                gridIndex: 1
            }],
            yAxis: [{
                splitLine: {show: false}
            }, {
                splitLine: {show: false},
                gridIndex: 1
            }],
            grid: [{
                bottom: '15%'
            }, {
                top: '30%'
            }],
            series: [{
                type: 'line',
                showSymbol: false,
                data: y
            }]
        };
    }

    getScatterOption(scatterData) {
        this.ScatterOption = {
            type: 'value',
            xAxis: {},
            yAxis: {},
            series: [{
                symbolSize: 7,
                data: scatterData,
                type: 'scatter'
            }],
            color: ['blue']
        };

        return this.ScatterOption;
    }

    getBarOption() {
        this.BarOption = {
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                }
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            },
            dataset: {
                source: [
                    ['product', '2015', '2016', '2017'],
                    ['Matcha Latte', 43.3, 85.8, 93.7],
                    ['Milk Tea', 83.1, 73.4, 55.1],
                    ['Cheese Cocoa', 86.4, 65.2, 82.5],
                    ['Walnut Brownie', 72.4, 53.9, 39.1]
                ]
            },
            xAxis: [
                {
                    type: 'category',
                    // data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    axisTick: {
                        alignWithLabel: true
                    }
                }
            ],
            yAxis: [
                {
                    // type: 'value'
                }
            ],
            series: [
                {
                    // name: '直接访问',
                    type: 'bar',
                    // barWidth: '60%',
                    // data: [10, 52, 200, 334, 390, 330, 220]
                },
                { type: 'bar' },
                { type: 'bar' }
            ]
        };

        return this.BarOption;
    }
    getLineOption(xLineData, yLineData) {
        this.LineOption = {
            xAxis: {
                type: 'category',
                data: xLineData
            },
            yAxis: {
                type: 'value'
            },
            series: [{
                data: yLineData,
                type: 'line',
                smooth: true
            }]
        };
        return this.LineOption;
    }
    getPieOption() {
        this.PieOption = {
            tooltip: {
                trigger: 'item',
                formatter: '{a} <br/>{b}: {c} ({d}%)'
            },
            legend: {
                orient: 'vertical',
                x: 'left',
                data: ['Example1', 'Example2', 'Example3']
            },
            roseType: 'angle',
            series: [
                {
                    name: 'PieChart',
                    type: 'pie',
                    radius: [0, '50%'],
                    data: [
                        { value: 235, name: 'Example1' },
                        { value: 210, name: 'Example2' },
                        { value: 162, name: 'Example3' }
                    ]
                }
            ]
        }
        return this.PieOption;
    }

    getAnimationBarOption(xAxisData,data1,data2) {
        this.AnimationBarOption = {
            legend: {
                data: ['Actual Viewership', 'Predicted Viewership'],
                align: 'left'
            },
            /* toolbox: {
                // y: 'bottom',
                feature: {
                    magicType: {
                        type: ['stack', 'tiled']
                    },
                    dataView: {},
                    saveAsImage: {
                        pixelRatio: 2
                    }
                }
            }, */
            tooltip: {},
            xAxis: {
                data: xAxisData,
                silent: false,
                splitLine: {
                    show: true
                }
            },
            yAxis: {
            },
            series: [{
                name: 'Actual',
                type: 'bar',
                data: data1,
                animationDelay: function (idx) {
                    return idx * 10;
                }
            }, {
                name: 'Predicted',
                type: 'bar',
                data: data2,
                animationDelay: function (idx) {
                    return idx * 10 + 100;
                }
            }],
            animationEasing: 'elasticOut',
            animationDelayUpdate: function (idx) {
                return idx * 5;
            },
            color: ['red','yellow']
        };

        return this.AnimationBarOption;
    }
}
