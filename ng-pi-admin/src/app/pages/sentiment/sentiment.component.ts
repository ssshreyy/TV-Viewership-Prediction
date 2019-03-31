
import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { HttpHeaders } from '@angular/common/http';

@Component({
  selector: 'app-visual',
  templateUrl: './sentiment.component.html',
  styleUrls: ['./sentiment.component.scss']
})
export class SentimentComponent implements OnInit {

  serverData: JSON;
  analysedData: JSON;
  arrayOfKeys;
  pageSize = 10;
  pageNumber = 1;
  pre=false;
  show=false;

  // constructor() { }
  constructor(private httpClient: HttpClient) {
  }

  ngOnInit() {
  }

  showTweets(){
    this.httpClient.get("http://127.0.0.1:5003/show").subscribe((data) => {
      this.serverData = data as JSON;

      console.log(this.serverData)

      this.arrayOfKeys = Object.keys(this.serverData);
      
      // this.analysedData=this.serverData;
      this.show=true;
    })
  }

  sentimentTweets(){
    this.httpClient.get("http://127.0.0.1:5003/sentiment").subscribe((data2) => {
      this.analysedData = data2 as JSON;
      this.arrayOfKeys.forEach(element => {
        
      });
      this.pre=true;
      console.log(this.analysedData)
    })
  }

  pageChanged(pN: number): void {
    this.pageNumber = pN;
  }

}