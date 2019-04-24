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

    getGradientOption2(xLineData, yLineData, yLineData2) {
        return {
            title: {
                
            },
            tooltip: {
                trigger: 'axis'
            },
            legend: {
                data:['Actual','Predicted']
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            },
            toolbox: {
                feature: {
                    saveAsImage: {}
                }
            },
            xAxis: {
                type: 'category',
                boundaryGap: false,
                data: xLineData
            },
            yAxis: {
                type: 'value'
            },
            series: [
                {
                    name:'Actual Viewership',
                    type:'line',
                    data:yLineData
                },
                {
                    name:'Predicted Viewership',
                    type:'line',
                    data:yLineData2
                }
            ],
            color:['red','aqua']
        };
        

        // return {
            
        //     visualMap: [{
        //         show: false,
        //         type: 'continuous',
        //         seriesIndex: 0,
        //         min: 0,
        //         max: 15000000
        //     },
        //     {
        //         show: false,
        //         type: 'continuous',
        //         seriesIndex: 0,
        //         min: 0,
        //         max: 15000000
        //     }],
        
        //     tooltip: {
        //         trigger: 'axis'
        //     },
        //     legend: {
        //         data: ['Example1', 'Example2']
        //     },
        //     xAxis: [{
        //         name: 'Air Date',
        //         data: xLineData,
        //         splitLine: {show: true}
        //     }, {
        //         name: 'Air Date',
        //         data: xLineData,
        //         gridIndex: 1,
        //         splitLine: {show: true}
        //     }],
        //     yAxis: [{
        //         name:'Viewership',
        //         splitLine: {show: true}
        //     }, {
        //         // name:'Viewership',
        //         splitLine: {show: true},
        //         gridIndex: 1
        //     }],
        //     grid: [{
        //         bottom: '15%'
        //     }, {
        //         top: '30%'
        //     }],
        //     series: [
        //         {
        //             type: 'line',
        //             showSymbol: false,
        //             data: yLineData
        //         },
        //         {
        //             type:'line',
        //             showSymbol: false,
        //             data: yLineData2
        //         }
        //     ],
        //     color: ['#000080','aqua']
        // };
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
            tooltip: {},
            xAxis: {
                data: xAxisData,
                silent: false,
                splitLine: {
                    show: true
                },
                name:'Air Date'
            },
            yAxis: {
                name: 'Viewership'
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
