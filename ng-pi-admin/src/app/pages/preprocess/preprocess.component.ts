import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { HttpHeaders } from '@angular/common/http';

@Component({
  selector: 'app-visual',
  templateUrl: './preprocess.component.html',
  styleUrls: ['./preprocess.component.scss']
})
export class PreprocessComponent implements OnInit {

  serverData: JSON;
  preprocessedData: JSON;
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
      
      // this.preprocessedData=this.serverData;
      this.show=true;
    })
  }

  preprocessTweets(){
    this.httpClient.get("http://127.0.0.1:5003/preprocess").subscribe((data2) => {
      this.preprocessedData = data2 as JSON;
      this.pre=true;
      console.log(this.preprocessedData)
    })
  }

  pageChanged(pN: number): void {
    this.pageNumber = pN;
  }

}
