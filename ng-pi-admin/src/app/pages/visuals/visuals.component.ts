import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { HttpHeaders } from '@angular/common/http';
import { VisualsService } from './visuals.service';

@Component({
  selector: 'app-visual',
  templateUrl: './visuals.component.html',
  styleUrls: ['./visuals.component.scss'],
  providers: [VisualsService]
})
export class VisualsComponent implements OnInit {
  showloading: boolean = false;
  BarOption;
  LineOption;
  GradientOption;
  PieOption;
  ScatterOption;
  AnimationBarOption;
  serverData: JSON;
  arrayOfKeys;
  plot=0;
  argument:number[][];
  yearList=[2009,2010,2011,2012,2013,2014,2015,2016,2017];

  // constructor() { }
  // constructor(private httpClient: HttpClient) {}

  constructor(private chartsService: VisualsService, private httpClient: HttpClient) {
  }
  
  ngOnInit() {
  }

  scatterPredVsActual(){
    this.httpClient.get("http://127.0.0.1:5003/scatter").subscribe((data) => {
      this.serverData = data as JSON;
      console.log(this.serverData);
      this.arrayOfKeys = Object.keys(this.serverData)
      this.argument = [];
      this.arrayOfKeys.forEach(element => {
        this.argument.push([this.serverData[element].US_Viewers_In_Millions, this.serverData[element].Predicted_Viewership]);
      });
      console.log(this.argument);
      this.ScatterOption = this.chartsService.getScatterOption(this.argument);
      this.plot = 1;
    })
  }

  lineTweetsPerYear(){
    this.httpClient.get("http://127.0.0.1:5003/line1").subscribe((data) => {
      console.log(data);
      this.LineOption = this.chartsService.getLineOption(
        this.yearList,
        data
      )
    });
    this.plot=2;
  }

  lineImdb(){
    this.httpClient.get("http://127.0.0.1:5003/line2").subscribe((data) => {
      console.log(data);
      this.GradientOption = this.chartsService.getGradientOption(
        data['ep'],
        data['imdb']
      )
    });
    this.plot=3;
  }

  lineViews(){
    this.httpClient.get("http://127.0.0.1:5003/line3").subscribe((data) => {
      console.log(data);
      this.GradientOption = this.chartsService.getGradientOption2(
        data['ep'],
        data['views'],
        data['predicted']
      )
    });
    this.plot=5;
  }

  barPosNeg(){
    this.httpClient.get("http://127.0.0.1:5003/bar2").subscribe((data) => {
      // console.log(data);
      console.log(data)
      var pos=[], neg=[],year=[];
      this.arrayOfKeys = Object.keys(data);
      this.arrayOfKeys.forEach(element => {
        year[element]=data[element].Year;
        pos[element]=data[element].Pos;
        neg[element]=data[element].Neg;
      });
      this.BarOption = this.chartsService.getBarOption(
        year,
        pos,
        neg
      );

    });
   
    this.plot=6;
  }

  barActualVsPredicted(){
    this.httpClient.get("http://127.0.0.1:5003/bar").subscribe((data) => {
      this.serverData = data as JSON;
      console.log(this.serverData);
      this.arrayOfKeys = Object.keys(this.serverData)
      this.argument = [];
      // this.arrayOfKeys.forEach(element => {
      //   this.argument.push([this.serverData[element].US_Viewers_In_Millions, this.serverData[element].Predicted_Viewership], this.serverData[element]);
      // });
      var x=[], y=[], z=[];
      this.arrayOfKeys.forEach(element => {
        x.push(data[element].Air_Date)
        y.push(data[element].US_Viewers_In_Millions);
        z.push(data[element].Predicted_Viewership);
      });
      console.log(x,y,z);
      this.AnimationBarOption = this.chartsService.getAnimationBarOption(x,y,z);
      this.plot = 4;
    });
  }

  wordcloud(data){
    console.log(data);

    const httpOptions = {
      headers: new HttpHeaders({
        'Content-Type':  'application/x-www-form-urlencoded',
        'Authorization': 'my-auth-token'
      })
    };
    var body = "show=" + data.show + "&year=" + data.year;

    this.httpClient.post("http://127.0.0.1:5003/search", body, httpOptions).subscribe((data) => {
      this.serverData = data as JSON;
      // alert(JSON.stringify(this.serverData))
      console.log(this.serverData)
      // alert(typeof this.serverData)
      this.arrayOfKeys = Object.keys(this.serverData)
      // alert(this.arrayOfKeys)
    })
  }
}
