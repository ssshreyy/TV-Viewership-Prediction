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
                name: 'Air Date',
                data: x,
                splitLine: {show: true}
            }, {
                name: 'Air Date',
                data: x,
                gridIndex: 1,
                splitLine: {show: true}
            }],
            yAxis: [{
                splitLine: {show: true}
            }, {
                splitLine: {show: true},
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
        var markLineOpt = {
            animation: false,
            label: {
                normal: {
                    formatter: 'y =x',
                    textStyle: {
                        align: 'right'
                    }
                }
            },
            lineStyle: {
                normal: {
                    type: 'dashed'
                }
            },
            tooltip: {
                formatter: 'y = x'
            },
            data: [[{
                coord: [0, 0],
                symbol: 'none'
            }, {
                coord: [12000000, 12000000],
                symbol: 'none'
            }]]
        };

        this.ScatterOption = {
            type: 'value',
            xAxis: {
                name: 'Actual Viewership'
            },
            yAxis: {
                name: 'Predicted Viewership'

            },
            series: [{
                symbolSize: 6,
                data: scatterData,
                type: 'scatter',
                markLine: markLineOpt
            }],
            color: ['DeepSkyBlue']
        };

        return this.ScatterOption;
    }

    getBarOption(year,pos,neg) {
        this.BarOption ={
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross',
                    crossStyle: {
                        color: '#999'
                    }
                }
            },
            toolbox: {
                feature: {
                    dataView: {show: true, readOnly: false},
                    magicType: {show: true, type: ['line', 'bar']},
                    restore: {show: true},
                    saveAsImage: {show: true}
                }
            },
            legend: {
                data:['Positive','Negative']
            },
            xAxis: [
                {
                    type: 'category',
                    name: 'Year',
                    data: year,
                    axisPointer: {
                        type: 'shadow'
                    }
                }
            ],
            yAxis: [
                {
                    type: 'value',
                    name: 'Number of Tweets',
                    min: 0,
                    max: 160000,
                    interval: 10000,
                    axisLabel: {
                        formatter: '{value}'
                    }
                }
            ],
            series: [
                {
                    name:'Negative',
                    type:'bar',
                    data:neg
                },
                {
                    name:'Positive',
                    type:'bar',
                    data:pos
                }
            ],
            color: ['#DC143C','#32CD32']
        };

        // this.BarOption = {
        //     tooltip: {
        //         trigger: 'axis',
        //         axisPointer: {
        //             type: 'shadow'
        //         }
        //     },
        //     grid: {
        //         left: '3%',
        //         right: '4%',
        //         bottom: '3%',
        //         containLabel: true
        //     },
        //     dataset: {
        //         source: [
        //             ['Sentiment', 'Positive', 'Negative'],
        //             ['Matcha Latte', 43.3, 85.8],
        //             ['Milk Tea', 83.1, 73.4],
        //             ['Cheese Cocoa', 86.4, 65.2],
        //             ['Walnut Brownie', 72.4, 53.9]                    
        //         ]
        //     },
        //     xAxis: [
        //         {
        //             type: 'category',
        //             data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        //             axisTick: {
        //                 alignWithLabel: true
        //             }
        //         }
        //     ],
        //     yAxis: [
        //         {
        //             // type: 'value'
        //         }
        //     ],
        //     series: [
        //         {
        //             // name: '直接访问',
        //             type: 'bar',
        //             // barWidth: '60%',
        //             // data: [10, 52, 200, 334, 390, 330, 220]
        //         },
        //         { type: 'bar' },
        //         { type: 'bar' }
        //     ]
        // };

        // x.forEach(element => {
        //     this.AnimationBarOption.BarOption['dataset']['source'].append([x[element],y1[element],y2[element]])
        // });
        return this.BarOption;
    }
    getLineOption(xLineData, yLineData) {
        this.LineOption = {
            xAxis: {
                type: 'category',
                data: xLineData,
                splitLine: {show: true},
                name: 'Year'
            },
            yAxis: {
                name: 'Number of Tweets',
                type: 'value',
                splitLine: {show: true}
            },
            series: [{
                data: yLineData,
                type: 'line',
                smooth: true
            }],
            color: ['green']
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
            // toolbox: {
            //     // y: 'bottom',
            //     feature: {
            //         magicType: {
            //             type: ['stack', 'tiled']
            //         },
            //         dataView: {},
            //         saveAsImage: {
            //             pixelRatio: 2
            //         }
            //     }
            // }, 
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
            color: ['black','yellow']
        };

        return this.AnimationBarOption;
    }
}
