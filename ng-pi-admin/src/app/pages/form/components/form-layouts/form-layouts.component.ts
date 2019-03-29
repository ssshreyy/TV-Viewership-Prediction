import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { HttpHeaders } from '@angular/common/http';

@Component({
  selector: 'app-form-layouts',
  templateUrl: './form-layouts.component.html',
  styleUrls: ['./form-layouts.component.scss']
})

export class FormLayoutsComponent implements OnInit {

  serverData: JSON;
  arrayOfKeys;
  pageSize = 10;
  pageNumber = 1;

  // constructor(){}
  constructor(private httpClient: HttpClient) {
  }
  
  ngOnInit() {
  }

  searchTweet(tweet) {
    console.log(tweet);

    const httpOptions = {
      headers: new HttpHeaders({
        'Content-Type':  'application/x-www-form-urlencoded',
        'Authorization': 'my-auth-token'
      })
    };
    var body = "username=" + tweet.username + "&query=" + tweet.query + "&since=" + tweet.since + "&until=" + tweet.until + "&maxNo=" + tweet.maxNo +"&top=" + tweet.top;
    this.httpClient.post("http://127.0.0.1:5003/search", body, httpOptions).subscribe((data) => {
      this.serverData = data as JSON;

      console.log(this.serverData)

      this.arrayOfKeys = Object.keys(this.serverData)
      
    })
  }

  pageChanged(pN: number): void {
    this.pageNumber = pN;
  }

}
